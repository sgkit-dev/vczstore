from contextlib import contextmanager

from vcztools.utils import make_icechunk_storage
from zarr.core.sync import sync
from zarr.storage._common import make_store


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


def copy_store_to_icechunk(source, dest):
    """Copy a Zarr store to a new Icechunk store."""
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(dest)
    repo = Repository.create(icechunk_storage)

    with repo.transaction("main", message="create") as dest:
        copy_store(source, dest)


@contextmanager
def icechunk_transaction(file_or_url, branch, *, message="update"):
    """Open an Icechunk store in a transaction, then amend last commit on completion."""
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(file_or_url)
    repo = Repository.open(icechunk_storage)

    with transaction_amend(repo, branch, message=message) as store:
        yield store


@contextmanager
def transaction_amend(repo, branch, message):
    """Like Icechunk's `transaction` context manager, but using amend not commit."""
    session = repo.writable_session(branch)
    yield session.store
    # use amend to overwrite previous commit
    session.amend(message=message)
