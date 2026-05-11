from bio2zarr.zarr_utils import create_group_array, get_compressor_config
from vcztools.utils import array_dims, open_zarr

from vczstore.utils import transaction


def rechunk(
    vcz,
    variants_array_name,
    variants_chunk_size,
    *,
    backend_storage=None,
):
    """Rechunk a variants array with a larger variants chunk size that is
    an exact multiple of the existing chunk size."""

    # TODO: check pre-conditions

    with transaction(vcz, backend_storage=backend_storage, message="rechunk") as vcz:
        root = open_zarr(vcz, mode="r+", backend_storage=backend_storage)

        var = variants_array_name
        arr = root[var]
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
