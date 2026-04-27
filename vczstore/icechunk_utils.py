from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from zarr.core.sync import sync
from zarr.storage._common import make_store


def _split_azure_container_path(path, *, file_or_url):
    path_parts = [part for part in path.split("/") if part]
    if len(path_parts) == 0:
        raise ValueError(
            f"Azure Icechunk URLs must include a container name: {file_or_url}"
        )
    return path_parts[0], "/".join(path_parts[1:])


def _make_azure_storage(ic, file_or_url):
    parsed = urlparse(file_or_url)

    if parsed.scheme in ("az", "azure"):
        account = parsed.netloc
        if account == "":
            raise ValueError(
                "Azure Icechunk URLs must use the form "
                "'az://<account>/<container>/<prefix>': "
                f"{file_or_url}"
            )
        container, prefix = _split_azure_container_path(
            parsed.path, file_or_url=file_or_url
        )
    elif parsed.scheme in ("abfs", "abfss"):
        if "@" not in parsed.netloc:
            raise ValueError(
                "ABFS Icechunk URLs must use the form "
                "'abfs://<container>@<account>.dfs.core.windows.net/<prefix>': "
                f"{file_or_url}"
            )
        container, account_host = parsed.netloc.split("@", 1)
        account = account_host.removesuffix(".dfs.core.windows.net")
        prefix = parsed.path.lstrip("/")
    elif parsed.scheme == "https" and parsed.netloc.endswith(
        (".blob.core.windows.net", ".dfs.core.windows.net")
    ):
        account = parsed.netloc.removesuffix(".blob.core.windows.net").removesuffix(
            ".dfs.core.windows.net"
        )
        container, prefix = _split_azure_container_path(
            parsed.path, file_or_url=file_or_url
        )
    else:
        raise ValueError(f"Unsupported Azure URL for icechunk: {file_or_url}")

    return ic.azure_storage(
        account=account,
        container=container,
        prefix=prefix,
        from_env=True,
    )


def make_icechunk_storage(file_or_url):
    """Convert a file or URL to an Icechunk Storage object."""
    import icechunk as ic

    if isinstance(file_or_url, str):
        if "://" not in file_or_url:  # local path
            return ic.Storage.new_local_filesystem(file_or_url)
        elif file_or_url.startswith("s3://"):
            url_parsed = urlparse(file_or_url)
            return ic.s3_storage(
                bucket=url_parsed.netloc,
                prefix=url_parsed.path.lstrip("/"),
                from_env=True,
            )
        elif file_or_url.startswith(
            ("az://", "azure://", "abfs://", "abfss://", "https://")
        ):
            return _make_azure_storage(ic, file_or_url)
        else:
            raise ValueError(f"Unsupported URL for icechunk: {file_or_url}")
    elif isinstance(file_or_url, Path):
        path = file_or_url.resolve()  # make absolute
        return ic.Storage.new_local_filesystem(str(path))
    else:
        raise TypeError(f"Unsupported URL type for icechunk: {type(file_or_url)}")


def delete_previous_snapshots(repo, branch="main"):
    """
    Delete all previous snapshots except the current one
    to avoid retaining data that has been explicitly removed.
    """
    # see https://icechunk.io/en/stable/expiration/

    current_snapshot = list(repo.ancestry(branch=branch))[0]
    expiry_time = current_snapshot.written_at
    repo.expire_snapshots(older_than=expiry_time)
    repo.garbage_collect(expiry_time)


# inspired by commit f3c123d3a2a94b7f14bc995e3897ee6acc9acbd1 in zarr-python
def copy_store(source, dest):
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.testing.stateful import SyncStoreWrapper

    # ensure source and dest are both stores
    source = sync(make_store(source))
    dest = sync(make_store(dest))

    s = SyncStoreWrapper(source)
    d = SyncStoreWrapper(dest)
    # need reverse=True to create zarr.json before chunks (otherwise icechunk complains)
    for source_key in sorted(s.list(), reverse=True):
        buffer = s.get(source_key, default_buffer_prototype())
        d.set(source_key, buffer)


def copy_store_chunks(source, dest, array_key, chunk_offset):
    print("copy_store_chunks", array_key)
    import zarr
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
    from zarr.storage._utils import _relativize_path
    from zarr.testing.stateful import SyncStoreWrapper

    # ensure source and dest are both stores
    source = sync(make_store(source))
    dest = sync(make_store(dest))

    s = SyncStoreWrapper(source)
    d = SyncStoreWrapper(dest)

    source_root = zarr.open(source)
    dest_root = zarr.open(dest)

    for source_key in s.list():  # TODO: use prefix?
        try:
            chunk_part = _relativize_path(path=source_key, prefix=array_key)
            if chunk_part in (".zattrs", ".zarray"):  # TODO others - where to get list?
                continue
        except ValueError:
            continue

        source_metadata = source_root[array_key].metadata
        if source_metadata.zarr_format == 2:
            chunk_key_encoding = V2ChunkKeyEncoding(
                separator=source_metadata.dimension_separator
            )
        else:
            chunk_key_encoding = source_metadata.chunk_key_encoding
        source_chunk_coords = chunk_key_encoding.decode_chunk_key(chunk_part)

        dest_chunk_coords = tuple(
            c + co for c, co in zip(source_chunk_coords, chunk_offset)
        )
        dest_chunk_key = dest_root[array_key].metadata.encode_chunk_key(
            dest_chunk_coords
        )
        dest_key = f"{array_key}/{dest_chunk_key}"

        print(
            f"copying {source_key}, source_chunk_coords {source_chunk_coords} "
            f"to {dest_key}"
        )
        buffer = s.get(source_key, default_buffer_prototype())
        d.set(dest_key, buffer)


def copy_store_to_icechunk(source, dest):
    """Copy a Zarr store to a new Icechunk store."""
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(dest)
    repo = Repository.create(icechunk_storage)

    with repo.transaction("main", message="create") as dest:
        copy_store(source, dest)


@contextmanager
def icechunk_transaction(file_or_url, branch, *, message="update"):
    """Open an Icechunk store in a transaction, then commit on completion."""
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(file_or_url)
    repo = Repository.open(icechunk_storage)

    with repo.transaction(branch, message=message) as store:
        yield store
