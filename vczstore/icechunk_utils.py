from contextlib import contextmanager

from vcztools.utils import make_icechunk_storage

from vczstore.utils import copy_store


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
