from bio2zarr.zarr_utils import create_empty_group_array, get_compressor_config
from vcztools.utils import array_dims, make_icechunk_storage, open_zarr

from vczstore.utils import delete_previous_snapshots, transaction, variant_chunk_slices


def rechunk(
    vcz,
    variants_array_name,
    variants_chunk_size,
    *,
    backend_storage="icechunk",
):
    """Rechunk a variants array with a larger variants chunk size that is
    an exact multiple of the existing chunk size."""

    if backend_storage != "icechunk":
        raise ValueError("Only icechunk storage is supported for rechunk")

    # TODO: check pre-conditions

    var = variants_array_name
    var_rechunked = f"{var}_rechunked"
    var_delete = f"{var}_delete"

    # separate transactions for data and restructuring operations as they can't be mixed
    # https://icechunk.io/en/stable/guides/moving-nodes/#rearrange-sessions
    with transaction(vcz, backend_storage=backend_storage, message="rechunk") as store:
        root = open_zarr(store, mode="r+", backend_storage=backend_storage)
        variant_chunks_in_batch = variants_chunk_size // root[var].chunks[0]

        arr = root[var]

        new_chunks = (variants_chunk_size,) + arr.chunks[1:]

        create_empty_group_array(
            root,
            var_rechunked,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=new_chunks,
            compressor=get_compressor_config(arr),
            dimension_names=array_dims(arr),
        )
        arr_rechunked = root[var_rechunked]

        for v_sel in variant_chunk_slices(root, variant_chunks_in_batch):
            arr_rechunked[v_sel, ...] = arr[v_sel, ...]

    with transaction(
        vcz, backend_storage=backend_storage, rearrange=True, message="rechunk rename"
    ) as store:
        session = store.session
        session.move(f"/{var}", f"/{var_delete}")
        session.move(f"/{var_rechunked}", f"/{var}")

    with transaction(
        vcz, backend_storage=backend_storage, message="rechunk delete"
    ) as store:
        root = open_zarr(store, mode="r+", backend_storage=backend_storage)
        del root[var_delete]

    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(vcz)
    repo = Repository.open(icechunk_storage)
    delete_previous_snapshots(repo)
