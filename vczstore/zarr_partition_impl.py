import numpy as np
import zarr
from bio2zarr.vcz import VcfZarrPartition
from vcztools.utils import search
from vcztools.vcf_writer import dims

from vczstore.utils import missing_val


def remove_init(vcz, sample_id, num_partitions):
    root = zarr.open(vcz, mode="r+")
    all_samples = root["sample_id"][:]

    # find index of sample to remove
    unknown_samples = np.setdiff1d(sample_id, all_samples)
    if len(unknown_samples) > 0:
        raise ValueError(f"unrecognised sample: {sample_id}")
    selection = search(all_samples, sample_id)

    # overwrite sample data
    root["sample_id"][selection] = ""

    # store remove parameters in zarr attributes
    root.attrs["remove"] = {
        "num_partitions": num_partitions,
        "sample_index": int(selection),
    }


def remove_partition(vcz, partition_index):
    root = zarr.open(vcz, mode="r+")

    remove_attrs = root.attrs["remove"]
    num_partitions = int(remove_attrs["num_partitions"])
    sample_index = int(remove_attrs["sample_index"])

    n_variants = root["variant_position"].shape[0]
    variants_chunk_size = root["variant_position"].chunks[0]

    partitions = VcfZarrPartition.generate_partitions(
        n_variants, variants_chunk_size, num_partitions
    )
    partition = partitions[partition_index]

    # overwrite call variables
    for var in root.keys():
        arr = root[var]
        if (
            var.startswith("call_")
            and dims(arr)[0] == "variants"
            and dims(arr)[1] == "samples"
        ):
            # TODO: check chunk size of variable
            sl = slice(partition.start, partition.stop)
            root[var][sl, sample_index, ...] = missing_val(arr)


def remove_finalise(vcz):
    root = zarr.open(vcz, mode="r+")
    del root.attrs["remove"]
