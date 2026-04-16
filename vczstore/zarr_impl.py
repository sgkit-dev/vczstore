import logging

import numpy as np
import zarr
from bio2zarr.zarr_utils import create_group_array, get_compressor_config
from more_itertools import peekable
from vcztools.constants import STR_FILL, STR_MISSING
from vcztools.retrieval import variant_iter
from vcztools.utils import array_dims, search

from vczstore.utils import missing_val, variant_chunk_slices

logger = logging.getLogger(__name__)


def append(vcz1, vcz2):
    """Append vcz2 to vcz1 in place"""
    root1 = zarr.open(vcz1, mode="r+")
    root2 = zarr.open(vcz2, mode="r")

    # check preconditions
    n_variants1 = root1["variant_contig"].shape[0]
    n_variants2 = root2["variant_contig"].shape[0]
    if n_variants1 != n_variants2:
        raise ValueError(
            "Stores being appended must have same number of variants. "
            f"First has {n_variants1}, second has {n_variants2}"
        )
    for field in ("contig_id", "variant_contig", "variant_position", "variant_allele"):
        values1 = root1[field][:]
        values2 = root2[field][:]
        if np.any(values1 != values2):
            raise ValueError(
                f"Stores being appended must have same values for field '{field}'"
            )

    # append samples
    sample_id1 = root1["sample_id"]
    sample_id2 = root2["sample_id"]

    old_num_samples = sample_id1.shape[0]
    new_num_samples = old_num_samples + sample_id2.shape[0]
    new_shape = (new_num_samples,)
    sample_id1.resize(new_shape)
    sample_id1[old_num_samples:new_num_samples] = sample_id2[:]

    # resize genotype fields
    for var in root1.keys():
        if var.startswith("call_"):
            arr = root1[var]
            if arr.ndim == 2:
                new_shape = (arr.shape[0], new_num_samples)
                arr.resize(new_shape)
            elif arr.ndim == 3:
                new_shape = (arr.shape[0], new_num_samples, arr.shape[2])
                arr.resize(new_shape)
            else:
                raise ValueError("unsupported number of array_dims")

    # append genotype fields
    for variant_selection in variant_chunk_slices(root1):
        for var in root1.keys():
            if var.startswith("call_"):
                root1[var][variant_selection, old_num_samples:new_num_samples, ...] = (
                    root2[var][variant_selection, ...]
                )


def remove(vcz, sample_id):
    """Remove a sample from vcz and overwrite with missing data"""
    root = zarr.open(vcz, mode="r+")
    all_samples = root["sample_id"][:]

    # find index of sample to remove
    unknown_samples = np.setdiff1d(sample_id, all_samples)
    if len(unknown_samples) > 0:
        raise ValueError(f"unrecognised sample: {sample_id}")
    sample_selection = search(all_samples, sample_id)

    # overwrite sample data
    root["sample_id"][sample_selection] = ""
    for variant_selection in variant_chunk_slices(root):
        for var in root.keys():
            arr = root[var]
            if (
                var.startswith("call_")
                and array_dims(arr)[0] == "variants"
                and array_dims(arr)[1] == "samples"
            ):
                arr[variant_selection, sample_selection, ...] = missing_val(arr)


def normalise(vcz1, vcz2, vcz2_norm):
    """Normalise variants in vcz2 with respect to vcz1 and write to vcz2_norm"""
    index = index_variants(vcz1, vcz2)

    root1 = zarr.open(vcz1, mode="r")
    root2 = zarr.open(vcz2, mode="r")
    norm_root = zarr.open(vcz2_norm, mode="w", zarr_format=root2.metadata.zarr_format)

    n_variants = root1["variant_contig"].shape[0]
    norm_root.attrs.update(root2.attrs)

    for var in root2.keys():
        if var.startswith("call_"):
            # normalise genotype fields
            arr = root2[var]
            shape = (n_variants,) + arr.shape[1:]
            chunks = (root1[var].chunks[0],) + arr.chunks[1:]
            data = np.full(shape, fill_value=missing_val(arr), dtype=arr.dtype)
            data[index, ...] = arr[...]
            create_group_array(
                norm_root,
                var,
                data=data,
                shape=data.shape,
                dtype=data.dtype,
                compressor=get_compressor_config(arr),
                chunks=chunks,
                dimension_names=array_dims(arr),
            )
        else:
            # copy across from vcz1 or vcz2
            if var == "sample_id":
                arr = root2[var]
            else:
                arr = root1[var]
            create_group_array(
                norm_root,
                var,
                data=arr[...],
                shape=arr.shape,
                dtype=arr.dtype,
                compressor=get_compressor_config(arr),
                chunks=arr.chunks,
                dimension_names=array_dims(arr),
            )


def index_variants(vcz1, vcz2):
    """Return a boolean index for variants of vcz2 in vcz1"""
    root1 = zarr.open(vcz1, mode="r")
    root2 = zarr.open(vcz2, mode="r")

    if not np.all(root1["contig_id"][:] == root2["contig_id"][:]):
        raise ValueError("contig_id fields must be identical")

    n_variants = root1["variant_contig"].shape[0]
    index = np.zeros(n_variants, dtype=bool)

    fields = ["variant_contig", "variant_position", "variant_allele"]
    it1 = variant_iter(vcz1, fields=fields)
    it2 = peekable(variant_iter(vcz2, fields=fields))
    for i, variant in enumerate(it1):
        v = it2.peek(None)
        if v is None:
            # it2 is exhausted - leave rest of index as False
            break
        if variants_match(variant, v):
            # exact match - advance it2 and continue to next variant in it1
            next(it2)
            index[i] = True
        else:
            if cmp_variant_site(variant, v) > 0:
                raise ValueError(
                    f"Variant not in (or out of order in) first vcz: {variant_repr(v)}"
                )

    # when it1 is exhausted error if there are any variants left in it2
    v = it2.peek(None)
    if v is not None:
        raise ValueError(f"Variant not in first vcz: {variant_repr(v)}")

    return index


def cmp_variant_site(variant1, variant2):
    """Compare two variants using the contig and position fields"""
    c1 = variant1["variant_contig"]
    c2 = variant2["variant_contig"]
    if c1 != c2:
        return int(c1 > c2) - int(c1 < c2)
    p1 = variant1["variant_position"]
    p2 = variant2["variant_position"]
    return int(p1 > p2) - int(p1 < p2)


def variants_match(variant1, variant2):
    """Test if two variants have identical contig and position and equivalent alleles"""
    return (
        variant1["variant_contig"] == variant2["variant_contig"]
        and variant1["variant_position"] == variant2["variant_position"]
        and variant_alleles_are_equivalent(
            variant1["variant_allele"], variant2["variant_allele"]
        )
    )


def variant_alleles_are_equivalent(a, b):
    """Test if alleles are equivalent, ignoring padding"""

    def _strip_trailing(seq):
        lst = list(seq)
        while lst and lst[-1] in (STR_MISSING, STR_FILL):
            lst.pop()
        return lst

    return _strip_trailing(a) == _strip_trailing(b)


def variant_repr(variant):
    """Simple repr for a variant"""
    return (
        f"variant_contig={variant['variant_contig']}, "
        f"variant_position={variant['variant_position']}, "
        f"variant_allele={variant['variant_allele'].tolist()}"
    )
