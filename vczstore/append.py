import logging

import numpy as np
import zarr
from vcztools.utils import array_dims, open_zarr

from vczstore.utils import variant_chunk_slices, variants_progress

logger = logging.getLogger(__name__)


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


def _assert_append_arrays_compatible(name, arr1, arr2):
    dims1 = array_dims(arr1)
    dims2 = array_dims(arr2)
    if dims1 != dims2:
        raise ValueError(
            f"append requires {name!r} to have matching dimensions. "
            f"First has {dims1}, second has {dims2}"
        )
    if arr1.ndim != arr2.ndim:
        raise ValueError(
            f"append requires {name!r} to have matching number of dimensions. "
            f"First has {arr1.ndim}, second has {arr2.ndim}"
        )
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError(
            f"append requires {name!r} to have matching number of variants. "
            f"First has {arr1.shape[0]}, second has {arr2.shape[0]}"
        )
    if arr1.ndim > 2 and arr1.shape[2:] != arr2.shape[2:]:
        raise ValueError(
            f"append requires {name!r} to have matching trailing dimensions. "
            f"First has {arr1.shape[2:]}, second has {arr2.shape[2:]}"
        )


def append(vcz1, vcz2, *, show_progress=False, zarr_backend_storage=None):
    """Append vcz2 to vcz1 in place"""
    root1 = open_zarr(vcz1, mode="r+", zarr_backend_storage=zarr_backend_storage)
    root2 = zarr.open(vcz2, mode="r")  # assume local

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

    call_arrays = []
    for var in root1.keys():
        if var.startswith("call_"):
            if var not in root2:
                raise ValueError(
                    f"append requires {var!r} to be present in both stores"
                )
            arr1 = root1[var]
            arr2 = root2[var]
            _assert_append_arrays_compatible(var, arr1, arr2)
            call_arrays.append((var, arr1))
    _assert_variant_chunk_alignment(
        call_arrays,
        variant_chunk_size=root1["variant_contig"].chunks[0],
        operation="append",
    )
    _assert_variant_chunk_alignment(
        [(var, root2[var]) for var, _ in call_arrays],
        variant_chunk_size=root2["variant_contig"].chunks[0],
        operation="append",
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
    for _, arr in call_arrays:
        if arr.ndim == 2:
            new_shape = (arr.shape[0], new_num_samples)
            arr.resize(new_shape)
        elif arr.ndim == 3:
            new_shape = (arr.shape[0], new_num_samples, arr.shape[2])
            arr.resize(new_shape)
        else:
            raise ValueError("unsupported number of array_dims")

    # append genotype fields
    with variants_progress(n_variants1, "Append", show_progress) as pbar:
        for v_sel in variant_chunk_slices(root1):
            for var in root1.keys():
                if var.startswith("call_"):
                    root1[var][v_sel, old_num_samples:new_num_samples, ...] = root2[
                        var
                    ][v_sel, ...]
            pbar.update(v_sel.stop - v_sel.start)
