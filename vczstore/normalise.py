import asyncio

import numpy as np
import zarr
from bio2zarr.zarr_utils import (
    STRING_DTYPE_NAME,
    ZARR_FORMAT,
    StringDType,
    get_compressor_config,
    make_compressor,
    numcodecs,
)
from more_itertools import peekable
from vcztools.constants import STR_FILL, STR_MISSING
from vcztools.retrieval import VczReader
from vcztools.utils import array_dims, search
from zarr.api import asynchronous as zarr_async
from zarr.core.sync import sync

from vczstore.concurrency import resolve_io_concurrency, run_bounded
from vczstore.utils import (
    missing_val,
    variant_chunk_slices_for_array,
    variants_progress,
)


def normalise(
    vcz1, vcz2, vcz2_norm, show_progress=False, io_concurrency=None
):
    """Normalise variants in vcz2 with respect to vcz1 and write to vcz2_norm.

    vcz1, vcz2, vcz2_norm are paths or Zarr stores. Variants in vcz1 not present
    in vcz2 are filled with missing values.
    """
    resolved_io_concurrency = resolve_io_concurrency(io_concurrency)
    sync(
        _normalise_async(
            vcz1,
            vcz2,
            vcz2_norm,
            show_progress=show_progress,
            io_concurrency=resolved_io_concurrency,
        )
    )


async def _create_empty_group_array_async(
    group,
    name,
    *,
    shape,
    dtype,
    chunks,
    compressor=None,
    filters=None,
    dimension_names=None,
    **kwargs,
):
    new_kwargs = {**kwargs}
    new_kwargs.pop("zarr_format", None)
    if compressor is not None:
        new_kwargs["compressors"] = [make_compressor(compressor)]
    if ZARR_FORMAT == 2:
        codecs = []
        if filters is not None:
            codecs = [numcodecs.get_codec(f) for f in filters]
        if dtype == STRING_DTYPE_NAME or dtype == StringDType():
            codecs.append(numcodecs.VLenUTF8())
        if len(codecs) == 0:
            codecs = None
        new_kwargs["filters"] = codecs
    else:
        new_kwargs["dimension_names"] = dimension_names

    array = await group.create_array(
        name=name, shape=shape, dtype=dtype, chunks=chunks, **new_kwargs
    )
    if ZARR_FORMAT == 2 and dimension_names is not None:
        await array.update_attributes({"_ARRAY_DIMENSIONS": dimension_names})
    return array


def _assert_variant_chunk_alignment(arrays, *, variant_chunk_size, operation):
    for name, arr in arrays:
        dims = array_dims(arr)
        if dims is None or len(dims) == 0 or dims[0] != "variants":
            raise ValueError(f"{operation} requires {name!r} to be chunked on variants")
        if arr.chunks is None or arr.chunks[0] != variant_chunk_size:
            raise ValueError(
                f"{operation} requires {name!r} to use VCZ-aligned variant chunks of "
                f"size {variant_chunk_size}"
            )


async def _normalise_async(
    vcz1, vcz2, vcz2_norm, *, show_progress=False, io_concurrency
):
    index, remap_alleles, allele_mappings, updated_allele_mappings = (
        await asyncio.to_thread(index_variants, vcz1, vcz2, show_progress=show_progress)
    )

    if len(updated_allele_mappings) > 0:
        raise NotImplementedError(f"New alleles found: {updated_allele_mappings}")

    root1, root2 = await asyncio.gather(
        zarr_async.open_group(store=vcz1, mode="r"),
        zarr_async.open_group(store=vcz2, mode="r"),
    )
    norm_root = await zarr_async.open_group(
        store=vcz2_norm, mode="w", zarr_format=root2.metadata.zarr_format
    )

    root1_arrays = {name: arr async for name, arr in root1.arrays()}
    root2_arrays = [item async for item in root2.arrays()]

    variant_contig1 = await root1.getitem("variant_contig")
    n_variants = variant_contig1.shape[0]
    variants_chunk_size = variant_contig1.chunks[0]
    await norm_root.update_attributes(dict(root2.attrs))

    variant_arrays = []
    static_arrays = []
    variant_output_arrays = []

    for name, arr2 in root2_arrays:
        if name.startswith("call_"):
            for dim in array_dims(arr2):
                if dim in ("alleles", "alt_alleles", "genotypes") or dim.startswith(
                    "local_"
                ):
                    raise NotImplementedError(
                        f"Allele remapping not supported for dim {dim} in "
                        f"variable {name}"
                    )
            shape = (n_variants,) + arr2.shape[1:]
            chunks = (variants_chunk_size,) + arr2.chunks[1:]
            norm_arr = await _create_empty_group_array_async(
                norm_root,
                name,
                shape=shape,
                dtype=arr2.dtype,
                chunks=chunks,
                compressor=get_compressor_config(arr2),
                dimension_names=array_dims(arr2),
            )
            variant_arrays.append((name, arr2, norm_arr, True))
            variant_output_arrays.append((name, norm_arr))
        else:
            if name == "sample_id":
                source_arr = arr2
            else:
                source_arr = root1_arrays[name]
            norm_arr = await _create_empty_group_array_async(
                norm_root,
                name,
                shape=source_arr.shape,
                dtype=source_arr.dtype,
                chunks=source_arr.chunks,
                compressor=get_compressor_config(source_arr),
                dimension_names=array_dims(source_arr),
            )
            dims = array_dims(source_arr)
            if dims is not None and len(dims) > 0 and dims[0] == "variants":
                variant_arrays.append((name, source_arr, norm_arr, False))
                variant_output_arrays.append((name, norm_arr))
            else:
                static_arrays.append((source_arr, norm_arr))

    _assert_variant_chunk_alignment(
        variant_output_arrays,
        variant_chunk_size=variants_chunk_size,
        operation="normalise",
    )

    for source_arr, norm_arr in static_arrays:
        await norm_arr.setitem(slice(None), await source_arr.getitem(slice(None)))

    match_idx = np.where(index)[0]
    remap_idx = np.where(remap_alleles)[0]

    chunk_bounds = np.arange(0, n_variants, step=variants_chunk_size)
    chunk_bounds = np.append(chunk_bounds, [n_variants])
    match_starts = np.searchsorted(match_idx, chunk_bounds)
    remap_starts = np.searchsorted(remap_idx, chunk_bounds)
    allele_mappings_list = list(allele_mappings.values())

    variant_slices = list(variant_chunk_slices_for_array(variant_contig1))

    async def worker(item):
        i, v_sel = item
        for name, source_arr, norm_arr, is_call_array in variant_arrays:
            if is_call_array:
                chunk_n = v_sel.stop - v_sel.start
                shape = (chunk_n,) + source_arr.shape[1:]
                data = np.full(
                    shape,
                    fill_value=missing_val(source_arr),
                    dtype=source_arr.dtype,
                )

                match_sl = slice(match_starts[i], match_starts[i + 1])
                local_idx = match_idx[match_sl] - v_sel.start
                if local_idx.size > 0:
                    data[local_idx, ...] = await source_arr.getitem(match_sl)

                if name == "call_genotype":
                    remap_sl = slice(remap_starts[i], remap_starts[i + 1])
                    local_remap_idx = remap_idx[remap_sl] - v_sel.start
                    chunk_maps = allele_mappings_list[remap_sl]
                    remap_genotypes(data, local_remap_idx, chunk_maps)

                await norm_arr.setitem(v_sel, data)
            else:
                await norm_arr.setitem(v_sel, await source_arr.getitem(v_sel))

    with variants_progress(n_variants, "Write", show_progress) as pbar:
        await run_bounded(
            list(enumerate(variant_slices)),
            worker,
            max_concurrency=io_concurrency,
            on_item_done=lambda item: pbar.update(item[1].stop - item[1].start),
        )


def remap_genotypes(gt, indices, mappings):
    """Update a genotype array in-place by remapping allele indices.

    indices and mappings are parallel arrays of variant positions and their
    allele index remappings.
    """
    num_samples = gt.shape[1]
    ploidy = gt.shape[2]
    for i, mapping in zip(indices, mappings):
        for j in range(num_samples):
            for k in range(ploidy):
                val = gt[i, j, k]
                if val >= 0:
                    gt[i, j, k] = mapping[val]


def index_variants(vcz1, vcz2, *, show_progress=False):
    """Construct an index for variants of vcz2 that are in vcz1.

    Returns:
        index: bool array of length n_variants (vcz1), True where variant is in vcz2.
        remap_alleles: bool array of length n_variants (vcz1), True where alleles
            need remapping.
        allele_mappings: dict {variant_index: int_array} giving the allele index
            remapping.
        updated_allele_mappings: dict {variant_index: str_array} of the updated
            (merged) alleles for variants where vcz2 has extra alleles not in vcz1.

    Note that the allele mappings are dicts which only contain sites where there
    is a remapping. This is an efficient way to store allele mappings, since they
    are rare, and are not known ahead of time.
    """
    root1 = zarr.open(vcz1, mode="r")
    root2 = zarr.open(vcz2, mode="r")

    if not np.all(root1["contig_id"][:] == root2["contig_id"][:]):
        raise ValueError("contig_id fields must be identical")

    n_variants = root1["variant_contig"].shape[0]
    index = np.zeros(n_variants, dtype=bool)
    remap_alleles = np.zeros(n_variants, dtype=bool)
    allele_mappings = {}
    updated_allele_mappings = {}

    fields = ["variant_contig", "variant_position", "variant_allele"]
    it1 = VczReader(root1).variants(fields=fields)
    it2 = peekable(VczReader(root2).variants(fields=fields))

    with variants_progress(n_variants, "Index", show_progress) as pbar:
        for i, variant in enumerate(it1):
            v = it2.peek(None)
            if v is None:
                break
            matched, mapping, updated = variants_match(variant, v)
            if matched:
                next(it2)
                index[i] = True
                if mapping is not None:
                    remap_alleles[i] = True
                    allele_mappings[i] = mapping
                if updated is not None:
                    updated_allele_mappings[i] = updated
            else:
                if cmp_variant_site(variant, v) > 0:
                    raise ValueError(
                        "Variant in vcz2 not found in vcz1 (or vcz2 is out of order): "
                        f"{variant_repr(v)}"
                    )
            pbar.update()

    v = it2.peek(None)
    if v is not None:
        raise ValueError(f"Variant not in first vcz: {variant_repr(v)}")

    return index, remap_alleles, allele_mappings, updated_allele_mappings


def cmp_variant_site(variant1, variant2) -> int:
    """Compare two variants using the contig and position fields.

    The return value is negative if the first variant is before the second,
    zero if they have the same contig and position, and positive if
    the first variant is after the second.
    """
    c1 = variant1["variant_contig"]
    c2 = variant2["variant_contig"]
    if c1 != c2:
        return int(c1 > c2) - int(c1 < c2)
    p1 = variant1["variant_position"]
    p2 = variant2["variant_position"]
    return int(p1 > p2) - int(p1 < p2)


def variants_match(
    variant1, variant2
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Test if two variants have identical site (contig, position) and equivalent
    alleles.

    Returns (match, mapping, updated) - see variant_alleles_are_equivalent.
    """
    if cmp_variant_site(variant1, variant2) != 0:
        return False, None, None
    else:
        return variant_alleles_are_equivalent(
            variant1["variant_allele"], variant2["variant_allele"]
        )


def variant_alleles_are_equivalent(
    a, b
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Test if alleles a and b are equivalent, ignoring missing/fill padding.

    Returns (match, mapping, updated) where mapping is the allele index remapping
    from b into a's order, and updated is the full merged allele array when b has
    extra alleles not in a.
    """
    ref_a = a[0]
    ref_b = b[0]

    if ref_a != ref_b:
        return False, None, None

    def _remove_missing_or_fill(arr):
        return arr[(arr != STR_MISSING) & (arr != STR_FILL)]

    alt_a = _remove_missing_or_fill(a[1:])
    alt_b = _remove_missing_or_fill(b[1:])

    if np.all(alt_a == alt_b):
        return True, None, None

    if np.intersect1d(alt_a, alt_b).shape[0] > 0:
        new_alleles = np.setdiff1d(alt_b, alt_a)
        if new_alleles.shape[0] > 0:
            updated = np.append(_remove_missing_or_fill(a), new_alleles, axis=0)
            mapping = search(updated, _remove_missing_or_fill(b))
            return True, mapping, updated
        else:
            mapping = search(_remove_missing_or_fill(a), _remove_missing_or_fill(b))
            return True, mapping, None

    return False, None, None


def variant_repr(variant) -> str:
    """Simple repr for a variant."""
    return (
        f"variant_contig={variant['variant_contig']}, "
        f"variant_position={variant['variant_position']}, "
        f"variant_allele={variant['variant_allele'].tolist()}"
    )
