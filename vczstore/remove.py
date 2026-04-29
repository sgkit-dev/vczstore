import logging
from contextlib import nullcontext

import numpy as np
from vcztools.utils import array_dims, open_zarr, search

from vczstore.utils import missing_val, variant_chunk_slices, variants_progress

logger = logging.getLogger(__name__)


def _assert_variant_chunk_alignment(arrays, *, variant_chunk_size, operation):
    for name, arr in arrays:
        dims = array_dims(arr)
        if (
            dims is None
            or len(dims) < 2
            or dims[0] != "variants"
            or dims[1] != "samples"
        ):
            raise ValueError(
                f"{operation} requires {name!r} to use variants/samples dimensions"
            )
        if arr.chunks is None or arr.chunks[0] != variant_chunk_size:
            raise ValueError(
                f"{operation} requires {name!r} to use VCZ-aligned variant chunks of "
                f"size {variant_chunk_size}"
            )


def remove(
    vcz,
    sample_id,
    *,
    variant_chunks_in_batch=None,
    show_progress=False,
    zarr_backend_storage=None,
):
    """Remove a sample from vcz and overwrite with missing data"""

    if variant_chunks_in_batch is None:
        variant_chunks_in_batch = 10
    if variant_chunks_in_batch < 1:
        raise ValueError("variant_chunks_in_batch must be greater than or equal to 1")

    if zarr_backend_storage == "icechunk":
        from vczstore.icechunk_utils import icechunk_transaction

        cm = icechunk_transaction(vcz, "main", message="remove")
    else:
        cm = nullcontext(vcz)
    with cm as vcz:
        root = open_zarr(vcz, mode="r+", zarr_backend_storage=zarr_backend_storage)
        n_variants = root["variant_contig"].shape[0]
        all_samples = root["sample_id"][:]

        # find index of sample to remove
        unknown_samples = np.setdiff1d(sample_id, all_samples)
        if len(unknown_samples) > 0:
            raise ValueError(f"unrecognised sample: {sample_id}")
        sample_selection = search(all_samples, sample_id)

        target_arrays = [
            (name, arr) for name, arr in root.arrays() if name.startswith("call_")
        ]

        _assert_variant_chunk_alignment(
            target_arrays,
            variant_chunk_size=root["variant_contig"].chunks[0],
            operation="remove",
        )

        # overwrite sample data
        root["sample_id"][sample_selection] = ""
        with variants_progress(n_variants, "Remove", show_progress) as pbar:
            for v_sel in variant_chunk_slices(root, variant_chunks_in_batch):
                for var in root.keys():
                    arr = root[var]
                    if (
                        var.startswith("call_")
                        and array_dims(arr)[0] == "variants"
                        and array_dims(arr)[1] == "samples"
                    ):
                        arr[v_sel, sample_selection, ...] = missing_val(arr)
            pbar.update(v_sel.stop - v_sel.start)
