from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage._common import make_store

from vczstore.concurrency import resolve_io_concurrency, run_bounded

ZARR_METADATA_FILENAMES = frozenset(
    ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")
)


def is_metadata_key(key):
    return key.rsplit("/", 1)[-1] in ZARR_METADATA_FILENAMES


def split_metadata_and_data_keys(keys):
    ordered_keys = sorted(keys, reverse=True)
    metadata_keys = []
    data_keys = []
    for key in ordered_keys:
        if is_metadata_key(key):
            metadata_keys.append(key)
        else:
            data_keys.append(key)
    return metadata_keys, data_keys


async def _copy_one_key(source, dest, key, *, prototype):
    buffer = await source.get(key, prototype=prototype)
    if buffer is None:
        raise FileNotFoundError(key)
    await dest.set(key, buffer)


async def copy_store_async(source, dest, *, io_concurrency):
    source_keys = [key async for key in source.list()]
    metadata_keys, data_keys = split_metadata_and_data_keys(source_keys)
    prototype = default_buffer_prototype()

    for key in metadata_keys:
        await _copy_one_key(source, dest, key, prototype=prototype)

    await run_bounded(
        data_keys,
        lambda key: _copy_one_key(source, dest, key, prototype=prototype),
        max_concurrency=io_concurrency,
    )


def copy_store(source, dest, *, io_concurrency=None):
    source = sync(make_store(source))
    dest = sync(make_store(dest))
    resolved_io_concurrency = resolve_io_concurrency(io_concurrency)
    sync(copy_store_async(source, dest, io_concurrency=resolved_io_concurrency))
