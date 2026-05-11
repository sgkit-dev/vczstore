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
    variants_chunk_size,
    *,
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
