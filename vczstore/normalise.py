import logging

import numpy as np
import zarr
from bio2zarr.zarr_utils import create_empty_group_array, get_compressor_config
from vcztools.constants import INT_FILL, INT_MISSING, STR_FILL, STR_MISSING
from vcztools.utils import array_dims, open_zarr, search

from vczstore.utils import (
    copy_store,
    missing_val,
    progress_bar,
    variant_chunk_slices,
)

logger = logging.getLogger(__name__)


def normalise(
    vcz1,
    vcz2,
    vcz2_norm,
    *,
    haploid_contigs=None,
    allow_new_alleles=False,
    variant_chunks_in_batch=None,
    show_progress=False,
    backend_storage=None,
):
    """Normalise variants in vcz2 with respect to vcz1 and write to vcz2_norm.

    vcz1, vcz2, vcz2_norm are paths or Zarr stores. Variants in vcz1 not present
    in vcz2 are filled with missing values.

    haploid_contigs is a list of contig IDs for haploid contigs, which defaults
    to ["X", "Y", "MT"]
    """

    if variant_chunks_in_batch is None:
        variant_chunks_in_batch = 10
    if variant_chunks_in_batch < 1:
        raise ValueError("variant_chunks_in_batch must be greater than or equal to 1")

    index, remap_alleles, update_alleles, allele_mappings, updated_allele_mappings = (
        index_variants(vcz1, vcz2, show_progress=show_progress)
    )

    if not allow_new_alleles and len(updated_allele_mappings) > 0:
        raise NotImplementedError(
            "New alleles found and `--allow-new-alleles` not specified: "
            f"{updated_allele_mappings}"
        )

    root1 = open_zarr(vcz1, mode="r", backend_storage=backend_storage)
    # assume vcz2, vcz2_norm are local
    root2 = zarr.open(vcz2, mode="r")
    norm_root = zarr.open(vcz2_norm, mode="w", zarr_format=root2.metadata.zarr_format)

    n_variants = root1["variant_contig"].shape[0]
    variants_chunk_size = root1["variant_contig"].chunks[0]
    norm_root.attrs.update(root2.attrs)

    if haploid_contigs is None:
        haploid_contigs = ["X", "Y", "MT"]

    if "call_genotype" in root2:
        ploidy = root2["call_genotype"].shape[2]
        contig_id = root1["contig_id"][:]
        contig_ploidy = np.array(
            [
                1 if contig_id[i] in haploid_contigs else ploidy
                for i in range(contig_id.shape[0])
            ]
        )
    else:
        contig_ploidy = None

    # create empty arrays
    for var in root2.keys():
        if var.startswith("call_"):
            # normalise genotype fields
            arr = root2[var]
            for dim in array_dims(arr):
                if dim in ("alleles", "alt_alleles", "genotypes") or dim.startswith(
                    "local_"
                ):
                    raise NotImplementedError(
                        f"Allele remapping not supported for dim {dim} in "
                        f"variable {var}"
                    )
            shape = (n_variants,) + arr.shape[1:]
            chunks = (variants_chunk_size,) + arr.chunks[1:]
            create_empty_group_array(
                norm_root,
                var,
                shape=shape,
                dtype=arr.dtype,
                chunks=chunks,
                compressor=get_compressor_config(arr),
                dimension_names=array_dims(arr),
            )
        else:
            # copy across from vcz1 or vcz2
            if var == "sample_id":
                arr = root2[var]
            else:
                arr = root1[var]
            create_empty_group_array(
                norm_root,
                var,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=arr.chunks,
                compressor=get_compressor_config(arr),
                dimension_names=array_dims(arr),
            )

    # copy direct from vcz1
    vcz1_copy_vars = [
        var
        for var in root1.keys()
        if not var.startswith("call_") and var != "sample_id"
    ]
    copy_store(vcz1, vcz2_norm, array_keys=vcz1_copy_vars)

    # copy direct from vcz2
    vcz2_copy_vars = ["sample_id"]
    copy_store(vcz2, vcz2_norm, array_keys=vcz2_copy_vars)

    # variables to normalise
    norm_vars = [var for var in root2.keys() if var.startswith("call_")]

    # turn bool indexes into int array indexes
    match_idx = np.where(index)[0]
    remap_idx = np.where(remap_alleles)[0]
    update_idx = np.where(update_alleles)[0]

    # find chunk boundaries
    chunk_bounds = np.arange(
        0, n_variants, step=variants_chunk_size * variant_chunks_in_batch
    )
    chunk_bounds = np.append(chunk_bounds, [n_variants])

    # find chunk offsets for indexes
    match_starts = np.searchsorted(match_idx, chunk_bounds)
    remap_starts = np.searchsorted(remap_idx, chunk_bounds)
    update_starts = np.searchsorted(update_idx, chunk_bounds)

    allele_mappings_list = list(allele_mappings.values())
    updated_allele_mappings_list = list(updated_allele_mappings.values())

    with progress_bar(n_variants, "Write", show_progress) as pbar:
        for i, v_sel in enumerate(variant_chunk_slices(root1, variant_chunks_in_batch)):
            for var, arr in root2.arrays():
                if var not in norm_vars:
                    continue

                arr = root2[var]
                chunk_n = v_sel.stop - v_sel.start
                shape = (chunk_n,) + arr.shape[1:]

                if var == "call_genotype":
                    data = np.full(shape, fill_value=INT_MISSING, dtype=arr.dtype)
                    if contig_ploidy is not None:
                        variant_ploidy = contig_ploidy[root1["variant_contig"][v_sel]]
                        for ploidy in range(1, data.shape[-1]):
                            ploidy_mask = variant_ploidy == ploidy
                            data[ploidy_mask, :, ploidy:] = INT_FILL
                    match_sl = slice(match_starts[i], match_starts[i + 1])
                    local_idx = match_idx[match_sl] - v_sel.start
                    data[local_idx, ...] = arr[match_sl, ...]

                    remap_sl = slice(remap_starts[i], remap_starts[i + 1])
                    local_remap_idx = remap_idx[remap_sl] - v_sel.start
                    chunk_maps = allele_mappings_list[remap_sl]
                    remap_genotypes(data, local_remap_idx, chunk_maps)

                else:
                    data = np.full(shape, fill_value=missing_val(arr), dtype=arr.dtype)
                    match_sl = slice(match_starts[i], match_starts[i + 1])
                    local_idx = match_idx[match_sl] - v_sel.start
                    data[local_idx, ...] = arr[match_sl, ...]

                norm_root[var][v_sel] = data

            # update variant_allele if needed
            if len(update_idx) > 0:
                var = "variant_allele"
                data = root1[var][v_sel, :]
                update_sl = slice(update_starts[i], update_starts[i + 1])
                local_update_idx = update_idx[update_sl] - v_sel.start
                chunk_maps = updated_allele_mappings_list[update_sl]
                update_variant_alleles(data, local_update_idx, chunk_maps)
                norm_root[var][v_sel] = data

            pbar.update(v_sel.stop - v_sel.start)

    if len(update_idx) > 0:
        norm_root["variant_allele"].attrs["normalise_new_alleles"] = True


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


def update_variant_alleles(variant_allele, indices, mappings):
    """Update a variant allele array in-place with new alleles.

    indices and mappings are parallel arrays of variant positions and their
    updated alleles.
    """
    for i, updated_alleles in zip(indices, mappings):
        variant_allele[i, : updated_alleles.shape[0]] = updated_alleles


def index_variants(vcz1, vcz2, *, show_progress=False, backend_storage=None):
    """Construct an index for variants of vcz2 that are in vcz1.

    Returns:
        index: bool array of length n_variants (vcz1), True where variant is in vcz2.
        remap_alleles: bool array of length n_variants (vcz1), True where alleles
            need remapping.
        update_alleles: bool array of length n_variants (vcz1), True where alleles
            need updating.
        allele_mappings: dict {variant_index: int_array} giving the allele index
            remapping.
        updated_allele_mappings: dict {variant_index: str_array} of the updated
            (merged) alleles for variants where vcz2 has extra alleles not in vcz1.

    Note that the allele mappings are dicts which only contain sites where there
    is a remapping. This is an efficient way to store allele mappings, since they
    are rare, and are not known ahead of time.
    """
    root1 = open_zarr(vcz1, mode="r", backend_storage=backend_storage)
    root2 = zarr.open(vcz2, mode="r")  # assume local

    contig_id1 = root1["contig_id"][:]
    contig_id2 = root2["contig_id"][:]

    if not np.all(contig_id1 == contig_id2):
        raise ValueError("contig_id fields must be identical")

    with progress_bar(6, "Index", show_progress, unit="array") as pbar:
        variant_contig1 = root1["variant_contig"][:]
        pbar.update()
        variant_position1 = root1["variant_position"][:]
        pbar.update()
        variant_allele1 = root1["variant_allele"][:]
        pbar.update()
        variant_contig2 = root2["variant_contig"][:]
        pbar.update()
        variant_position2 = root2["variant_position"][:]
        pbar.update()
        variant_allele2 = root2["variant_allele"][:]
        pbar.update()

    n_variants1 = variant_contig1.shape[0]
    logger.debug(f"index_variants: loaded {n_variants1} variants from vcz1")

    n_variants2 = variant_contig2.shape[0]
    logger.debug(f"index_variants: loaded {n_variants2} variants from vcz2")

    index = np.zeros(n_variants1, dtype=bool)
    remap_alleles = np.zeros(n_variants1, dtype=bool)
    update_alleles = np.zeros(n_variants1, dtype=bool)
    allele_mappings = {}
    updated_allele_mappings = {}

    # Encode (contig, position) as a single int64 key for vectorized site matching.
    pos_stride = int(max(variant_position1.max(), variant_position2.max())) + 1
    max_contig = int(max(variant_contig1.max(), variant_contig2.max()))
    if max_contig * pos_stride > np.iinfo(np.int64).max:
        raise ValueError(
            f"Cannot encode variants as int64 keys: "
            f"max_contig={max_contig} * pos_stride={pos_stride} overflows int64"
        )
    keys1 = variant_contig1.astype(np.int64) * pos_stride + variant_position1.astype(
        np.int64
    )
    keys2 = variant_contig2.astype(np.int64) * pos_stride + variant_position2.astype(
        np.int64
    )

    # For each vcz2 variant, find its matching site group in vcz1.
    left1_for_2 = np.searchsorted(keys1, keys2, side="left")
    right1_for_2 = np.searchsorted(keys1, keys2, side="right")
    group_size_in_1 = right1_for_2 - left1_for_2

    # For each vcz2 variant, find its own site group size.
    left2 = np.searchsorted(keys2, keys2, side="left")
    right2 = np.searchsorted(keys2, keys2, side="right")
    group_size_in_2 = right2 - left2

    # Simple sites: exactly 1 variant per site in both stores. Matching is trivial.
    is_simple = (group_size_in_1 == 1) & (group_size_in_2 == 1)
    simple_i2 = np.where(is_simple)[0]
    simple_i1 = left1_for_2[is_simple]

    # Vectorized allele comparison for simple sites (the common fast path).
    if variant_allele1.shape[1] == variant_allele2.shape[1] and len(simple_i1) > 0:
        exact_simple = np.all(
            variant_allele1[simple_i1] == variant_allele2[simple_i2], axis=1
        )
    else:
        exact_simple = np.zeros(len(simple_i1), dtype=bool)

    index[simple_i1[exact_simple]] = True

    # Python fallback for simple sites whose alleles aren't byte-identical.
    for j in np.where(~exact_simple)[0]:
        i1 = int(simple_i1[j])
        i2 = int(simple_i2[j])
        matched, mapping, updated = variant_alleles_are_equivalent(
            variant_allele1[i1], variant_allele2[i2]
        )
        if not matched:
            raise ValueError(
                "Variant alleles in vcz2 are not equivalent to vcz1. "
                f"vcz1: {variant_repr(contig_id1, variant_contig1, variant_position1, variant_allele1, i1)} "  # noqa E501
                f"vcz2: {variant_repr(contig_id2, variant_contig2, variant_position2, variant_allele2, i2)}"  # noqa E501
            )
        index[i1] = True
        if mapping is not None:
            remap_alleles[i1] = True
            allele_mappings[i1] = mapping
        if updated is not None:
            update_alleles[i1] = True
            updated_allele_mappings[i1] = updated

    # Multi-allelic sites: two-pointer allele matching within each site group.
    for site_key in np.unique(keys2[~is_simple]):
        i1_start = int(np.searchsorted(keys1, site_key, side="left"))
        i1_end = int(np.searchsorted(keys1, site_key, side="right"))
        i2_start = int(np.searchsorted(keys2, site_key, side="left"))
        i2_end = int(np.searchsorted(keys2, site_key, side="right"))

        if i1_start == i1_end:
            raise ValueError(
                "Variant in vcz2 not found in vcz1 (or vcz2 is out of order): "
                f"{variant_repr(contig_id2, variant_contig2, variant_position2, variant_allele2, i2_start)}"  # noqa E501
            )

        i1_ptr = i1_start
        for i2 in range(i2_start, i2_end):
            matched_this = False
            while i1_ptr < i1_end:
                matched, mapping, updated = variant_alleles_are_equivalent(
                    variant_allele1[i1_ptr], variant_allele2[i2]
                )
                if matched:
                    index[i1_ptr] = True
                    if mapping is not None:
                        remap_alleles[i1_ptr] = True
                        allele_mappings[i1_ptr] = mapping
                    if updated is not None:
                        update_alleles[i1_ptr] = True
                        updated_allele_mappings[i1_ptr] = updated
                    i1_ptr += 1
                    matched_this = True
                    break
                i1_ptr += 1
            if not matched_this:
                raise ValueError(
                    "Variant in vcz2 not found in vcz1 (or vcz2 is out of order): "
                    f"{variant_repr(contig_id2, variant_contig2, variant_position2, variant_allele2, i2)}"  # noqa E501
                )

    return (
        index,
        remap_alleles,
        update_alleles,
        allele_mappings,
        updated_allele_mappings,
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


def variant_repr(contig_id, variant_contig, variant_position, variant_allele, i) -> str:
    """Simple repr for a variant"""
    return (
        f"contig={contig_id[variant_contig[i]]}, "
        f"variant_position={variant_position[i]}, "
        f"variant_allele={variant_allele[i].tolist()}"
    )
