from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from vczstore.store_copy import copy_store as _copy_store


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

def copy_store(source, dest, *, io_concurrency=None):
    _copy_store(source, dest, io_concurrency=io_concurrency)


def copy_store_to_icechunk(source, dest, *, io_concurrency=None):
    """Copy a Zarr store to a new Icechunk store."""
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(dest)
    repo = Repository.create(icechunk_storage)

    with repo.transaction("main", message="create") as dest:
        copy_store(source, dest, io_concurrency=io_concurrency)


@contextmanager
def icechunk_transaction(file_or_url, branch, *, message="update"):
    """Open an Icechunk store in a transaction, then commit on completion."""
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(file_or_url)
    repo = Repository.open(icechunk_storage)

    with repo.transaction(branch, message=message) as store:
        yield store
