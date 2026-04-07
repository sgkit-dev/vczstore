import numpy as np
import zarr
from bio2zarr.vcz import VcfZarrPartition
from vcztools.utils import search
from vcztools.vcf_writer import dims

from vczstore.utils import missing_val


def append_init(vcz1, vcz2, num_partitions):
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
    # TODO: check these preconditions in each partition
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
                raise ValueError("unsupported number of dims")

    # store append parameters in zarr attributes
    root1.attrs["append"] = {
        "num_partitions": num_partitions,
        "old_num_samples": old_num_samples,
        "new_num_samples": new_num_samples,
    }


def append_partition(vcz1, vcz2, partition_index):
    root1 = zarr.open(vcz1, mode="r+")
    root2 = zarr.open(vcz2, mode="r")

    append_attrs = root1.attrs["append"]
    num_partitions = int(append_attrs["num_partitions"])
    old_num_samples = int(append_attrs["old_num_samples"])
    new_num_samples = int(append_attrs["new_num_samples"])

    n_variants = root1["variant_position"].shape[0]
    variants_chunk_size = root1["variant_position"].chunks[0]

    partitions = VcfZarrPartition.generate_partitions(
        n_variants, variants_chunk_size, num_partitions
    )
    partition = partitions[partition_index]

    # append genotype fields
    for var in root1.keys():
        if var.startswith("call_"):
            arr = root1[var]
            # TODO: check chunk size of variable
            sl = slice(partition.start, partition.stop)
            if arr.ndim == 2:
                arr[sl, old_num_samples:new_num_samples] = root2[var][sl, ...]
            elif arr.ndim == 3:
                arr[sl, old_num_samples:new_num_samples, :] = root2[var][sl, ...]
            else:
                raise ValueError("unsupported number of dims")


def append_finalise(vcz1, vcz2):
    root = zarr.open(vcz1, mode="r+")
    del root.attrs["append"]


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
