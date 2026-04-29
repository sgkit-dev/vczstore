from contextlib import contextmanager

import tqdm
from bio2zarr.vcf_utils import ceildiv
from vcztools.constants import FLOAT32_MISSING, INT_MISSING, STR_MISSING
from vcztools.utils import make_icechunk_storage
from zarr.core.sync import sync
from zarr.storage._common import make_store


def missing_val(arr):
    if arr.dtype.kind == "i":
        return INT_MISSING
    elif arr.dtype.kind == "f":
        return FLOAT32_MISSING
    elif arr.dtype.kind in ("O", "U", "T"):
        return STR_MISSING
    elif arr.dtype.kind == "b":
        return False
    else:
        raise ValueError(f"unrecognised dtype: {arr.dtype}")


def variant_chunk_slices(root, variant_chunks_in_batch=1):
    """A generator returning chunk slices along the variants dimension."""
    pos = root["variant_position"]
    size = pos.shape[0]
    v_chunksize = pos.chunks[0] * variant_chunks_in_batch
    for v_chunk in range(ceildiv(size, v_chunksize)):
        start = v_chunksize * v_chunk
        end = min(v_chunksize * (v_chunk + 1), size)
        yield slice(start, end)


def progress_bar(total, title, show_progress=False, unit="vars"):
    return tqdm.tqdm(
        total=total,
        desc=f"{title:>8}",
        unit_scale=True,
        unit=unit,
        smoothing=0.1,
        disable=not show_progress,
    )


# inspired by commit f3c123d3a2a94b7f14bc995e3897ee6acc9acbd1 in zarr-python
def copy_store(source, dest, array_keys=None):
    from zarr.core.buffer.core import default_buffer_prototype
    from zarr.testing.stateful import SyncStoreWrapper

    # ensure source and dest are both stores
    source = sync(make_store(source))
    dest = sync(make_store(dest))

    s = SyncStoreWrapper(source)
    d = SyncStoreWrapper(dest)
    # need reverse=True to create zarr.json before chunks (otherwise icechunk complains)
    for source_key in sorted(s.list(), reverse=True):
        if array_keys is not None and source_key.split("/")[0] not in array_keys:
            continue
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
