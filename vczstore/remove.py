import logging

import numpy as np
import zarr
from vcztools.utils import array_dims, search

from vczstore.utils import missing_val, variant_chunk_slices

logger = logging.getLogger(__name__)


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
