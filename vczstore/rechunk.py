import math

from bio2zarr.zarr_utils import create_group_array, get_compressor_config
from vcztools.utils import array_dims, open_zarr

from vczstore.utils import (
    compute_min_variants_chunk_size,
    has_variants_axis,
    transaction,
)


def rechunk(
    vcz,
    variants_array_name,
    variants_chunk_size=None,
    *,
    target_uncompressed_size_bytes=None,
    backend_storage=None,
):
    """Rechunk a variants array with a larger variants chunk size that is
    an exact multiple of the min variants chunk size."""

    with transaction(vcz, backend_storage=backend_storage, message="rechunk") as vcz:
        root = open_zarr(vcz, mode="r+", backend_storage=backend_storage)

        var = variants_array_name
        arr = root[var]

        if not has_variants_axis(arr):
            raise ValueError(
                f"Array '{var}' does not have variants as its first dimension"
            )

        min_chunk_size = compute_min_variants_chunk_size(root)

        if target_uncompressed_size_bytes is not None:
            variants_chunk_size = compute_variants_chunk_size_from_target(
                arr, target_uncompressed_size_bytes, min_chunk_size
            )

        if variants_chunk_size % min_chunk_size != 0:
            raise ValueError(
                f"variants_chunk_size={variants_chunk_size} is not an exact multiple "
                f"of the minimum variants chunk size {min_chunk_size}"
            )

        new_chunks = (variants_chunk_size,) + arr.chunks[1:]

        # read entire array into memory
        data = arr[:]

        # delete the array
        del root[var]

        # recreate the array with new chunk size
        create_group_array(
            root,
            var,
            data=data,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=new_chunks,
            compressor=get_compressor_config(arr),
            dimension_names=array_dims(arr),
        )


def compute_variants_chunk_size_from_target(
    arr, target_bytes: int, min_chunk_size: int
) -> int:
    """Compute the variants chunk size giving chunks of approximately
    target_bytes uncompressed."""
    bytes_per_variant_chunk = math.prod(arr.chunks[1:]) * arr.dtype.itemsize
    n = target_bytes // bytes_per_variant_chunk
    return max((n // min_chunk_size) * min_chunk_size, min_chunk_size)
