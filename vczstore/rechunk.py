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


def has_variants_axis(arr) -> bool:
    """Whether ``arr``'s first dimension is the variants axis."""
    dims = array_dims(arr)
    return dims is not None and len(dims) > 0 and dims[0] == "variants"


def compute_min_variants_chunk_size(root) -> int:
    """Compute the minimum variants-axis chunk size in a VCZ root.

    By spec ``call_*`` fields define the floor; every variant-only
    field must use a chunk size that is a positive integer multiple of
    it. Two ``call_*`` fields with different chunk sizes are a writer
    bug and raise ``ValueError`` here. When no ``call_*`` field is
    present, falls back to the minimum chunk size across variant-axis
    fields.
    """
    call_sizes: dict[str, int] = {}
    other_sizes: list[int] = []
    for name in root.array_keys():
        arr = root[name]
        if not has_variants_axis(arr):
            continue
        chunk_size = int(arr.chunks[0])
        if name.startswith("call_"):
            call_sizes[name] = chunk_size
        else:
            other_sizes.append(chunk_size)
    if len(call_sizes) > 0:
        sizes_set = set(call_sizes.values())
        if len(sizes_set) > 1:
            raise ValueError(
                f"call_* fields must share a single variants chunk size; "
                f"found {call_sizes}"
            )
        return next(iter(sizes_set))
    if len(other_sizes) > 0:
        return min(other_sizes)
    raise ValueError("no variant-axis fields in store")
