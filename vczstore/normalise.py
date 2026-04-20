import numpy as np
import zarr
from bio2zarr.zarr_utils import create_empty_group_array, get_compressor
from more_itertools import peekable
from vcztools.constants import STR_FILL, STR_MISSING
from vcztools.retrieval import VczReader
from vcztools.utils import array_dims, search

from vczstore.utils import missing_val, variant_chunk_slices


def normalise(vcz1, vcz2, vcz2_norm):
    """Normalise variants in vcz2 with respect to vcz1 and write to vcz2_norm.

    vcz1, vcz2, vcz2_norm are paths or Zarr stores. Variants in vcz1 not present
    in vcz2 are filled with missing values.
    """
    index, remap_alleles, allele_mappings, updated_allele_mappings = index_variants(
        vcz1, vcz2
    )

    if len(updated_allele_mappings) > 0:
        raise NotImplementedError(f"New alleles found: {updated_allele_mappings}")

    root1 = zarr.open(vcz1, mode="r")
    root2 = zarr.open(vcz2, mode="r")
    norm_root = zarr.open(vcz2_norm, mode="w", zarr_format=root2.metadata.zarr_format)

    n_variants = root1["variant_contig"].shape[0]
    variants_chunk_size = root1["variant_contig"].chunks[0]
    norm_root.attrs.update(root2.attrs)

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
                compressor=get_compressor(arr),
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
                compressor=get_compressor(arr),
                dimension_names=array_dims(arr),
            )

    # copy static (non-variant-axis) arrays
    for var, arr in root2.arrays():
        if array_dims(arr)[0] == "variants":
            continue
        # copy samples from vcz2, everything else from vcz1
        if var == "sample_id":
            arr = root2[var]
        else:
            arr = root1[var]
        norm_root[var][:] = arr[...]

    # turn bool indexes into int array indexes
    match_idx = np.where(index)[0]
    remap_idx = np.where(remap_alleles)[0]

    # find chunk boundaries
    chunk_bounds = np.arange(0, n_variants, step=variants_chunk_size)
    chunk_bounds = np.append(chunk_bounds, [n_variants])

    # find chunk offsets for indexes
    match_starts = np.searchsorted(match_idx, chunk_bounds)
    remap_starts = np.searchsorted(remap_idx, chunk_bounds)

    allele_mappings_list = list(allele_mappings.values())

    # copy variant-chunked arrays
    for i, v_sel in enumerate(variant_chunk_slices(root1)):
        for var, arr in root2.arrays():
            if array_dims(arr)[0] != "variants":
                continue
            # copy genotype fields from vcz2, everything else from vcz1
            if var.startswith("call_"):
                arr = root2[var]
                chunk_n = v_sel.stop - v_sel.start
                shape = (chunk_n,) + arr.shape[1:]
                data = np.full(shape, fill_value=missing_val(arr), dtype=arr.dtype)

                match_sl = slice(match_starts[i], match_starts[i + 1])
                local_idx = match_idx[match_sl] - v_sel.start
                data[local_idx, ...] = arr[match_sl, ...]

                if var == "call_genotype":
                    remap_sl = slice(remap_starts[i], remap_starts[i + 1])
                    local_remap_idx = remap_idx[remap_sl] - v_sel.start
                    chunk_maps = allele_mappings_list[remap_sl]
                    remap_genotypes(data, local_remap_idx, chunk_maps)

                norm_root[var][v_sel] = data
            else:
                arr = root1[var]
                norm_root[var][v_sel] = arr[v_sel, ...]


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


def index_variants(vcz1, vcz2):
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
    it1 = VczReader(vcz1).variants(fields=fields)
    it2 = peekable(VczReader(vcz2).variants(fields=fields))
    for i, variant in enumerate(it1):
        v = it2.peek(None)
        if v is None:
            # it2 is exhausted - leave rest of index as False
            break
        matched, mapping, updated = variants_match(variant, v)
        if matched:
            # advance it2 and continue to next variant in it1
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

    # when it1 is exhausted error if there are any variants left in it2
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

    Returns (match, mapping, updated) — see variant_alleles_are_equivalent.
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
    """Simple repr for a variant"""
    return (
        f"variant_contig={variant['variant_contig']}, "
        f"variant_position={variant['variant_position']}, "
        f"variant_allele={variant['variant_allele'].tolist()}"
    )
