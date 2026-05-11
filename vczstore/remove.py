import logging

import numpy as np
from vcztools.utils import array_dims, open_zarr, search

from vczstore.utils import (
    compute_min_variants_chunk_size,
    missing_val,
    progress_bar,
    transaction,
    variant_chunk_slices,
)

logger = logging.getLogger(__name__)


def remove(
    vcz,
    sample_id,
    *,
    variant_chunks_in_batch=None,
    show_progress=False,
    backend_storage=None,
):
    """Remove a sample from vcz and overwrite with missing data"""

    if variant_chunks_in_batch is None:
        variant_chunks_in_batch = 10
    if variant_chunks_in_batch < 1:
        raise ValueError("variant_chunks_in_batch must be greater than or equal to 1")

    with transaction(vcz, backend_storage=backend_storage, message="remove") as vcz:
        root = open_zarr(vcz, mode="r+", backend_storage=backend_storage)
        n_variants = root["variant_contig"].shape[0]
        all_samples = root["sample_id"][:]

        # check that all call_* fields have same variant chunk size
        compute_min_variants_chunk_size(root)

        # find index of sample to remove
        unknown_samples = np.setdiff1d(sample_id, all_samples)
        if len(unknown_samples) > 0:
            raise ValueError(f"unrecognised sample: {sample_id}")
        sample_selection = search(all_samples, sample_id)

        # overwrite sample data
        root["sample_id"][sample_selection] = ""
        with progress_bar(n_variants, "Remove", show_progress) as pbar:
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
