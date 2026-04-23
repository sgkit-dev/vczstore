import asyncio

import numpy as np
import pytest
import zarr

from vczstore.store_copy import (
    copy_store,
    copy_store_async,
    split_metadata_and_data_keys,
)


class FakeSourceStore:
    def __init__(self, data, delay=0.01, failing_key=None):
        self.data = data
        self.delay = delay
        self.failing_key = failing_key
        self.active_gets = 0
        self.max_active_gets = 0

    async def list(self):
        for key in self.data:
            yield key

    async def get(self, key, prototype):
        self.active_gets += 1
        self.max_active_gets = max(self.max_active_gets, self.active_gets)
        try:
            await asyncio.sleep(self.delay)
            if key == self.failing_key:
                raise RuntimeError(f"get failed for {key}")
            return prototype.buffer.from_bytes(self.data[key])
        finally:
            self.active_gets -= 1


class FakeDestStore:
    def __init__(self, metadata_keys, delay=0.01, failing_key=None):
        self.delay = delay
        self.failing_key = failing_key
        self.metadata_keys = set(metadata_keys)
        self.seen = []
        self.values = {}
        self.data_before_metadata = False

    async def set(self, key, value):
        if key not in self.metadata_keys and not self.metadata_keys.issubset(self.seen):
            self.data_before_metadata = True
        await asyncio.sleep(self.delay)
        if key == self.failing_key:
            raise RuntimeError(f"set failed for {key}")
        self.seen.append(key)
        self.values[key] = value.to_bytes()


def test_split_metadata_and_data_keys_puts_metadata_before_chunks():
    keys = ["sample/c/0", "sample/zarr.json", "zarr.json", "sample/c/1"]

    metadata_keys, data_keys = split_metadata_and_data_keys(keys)

    assert metadata_keys == ["zarr.json", "sample/zarr.json"]
    assert data_keys == ["sample/c/1", "sample/c/0"]


def test_copy_store_async_uses_bounded_concurrency_and_metadata_barrier():
    keys = ["sample/c/0", "sample/zarr.json", "zarr.json", "sample/c/1", "sample/c/2"]
    data = {key: key.encode() for key in keys}
    metadata_keys, _ = split_metadata_and_data_keys(keys)
    source = FakeSourceStore(data)
    dest = FakeDestStore(metadata_keys)

    asyncio.run(copy_store_async(source, dest, io_concurrency=2))

    assert source.max_active_gets <= 2
    assert dest.data_before_metadata is False
    assert dest.seen[: len(metadata_keys)] == metadata_keys
    assert dest.values == data


def test_copy_store_async_does_not_start_data_writes_after_metadata_failure():
    keys = ["sample/c/0", "sample/zarr.json", "zarr.json", "sample/c/1"]
    data = {key: key.encode() for key in keys}
    metadata_keys, _ = split_metadata_and_data_keys(keys)
    source = FakeSourceStore(data)
    dest = FakeDestStore(metadata_keys, failing_key="sample/zarr.json")

    with pytest.raises(RuntimeError, match="set failed"):
        asyncio.run(copy_store_async(source, dest, io_concurrency=2))

    assert dest.data_before_metadata is False
    assert "sample/c/0" not in dest.seen
    assert "sample/c/1" not in dest.seen


def test_copy_store_async_propagates_source_failures():
    keys = ["sample/c/0", "sample/zarr.json", "zarr.json", "sample/c/1"]
    data = {key: key.encode() for key in keys}
    metadata_keys, _ = split_metadata_and_data_keys(keys)
    source = FakeSourceStore(data, failing_key="sample/c/1")
    dest = FakeDestStore(metadata_keys)

    with pytest.raises(RuntimeError, match="get failed"):
        asyncio.run(copy_store_async(source, dest, io_concurrency=2))


def test_copy_store_async_propagates_destination_failures():
    keys = ["sample/c/0", "sample/zarr.json", "zarr.json", "sample/c/1"]
    data = {key: key.encode() for key in keys}
    metadata_keys, _ = split_metadata_and_data_keys(keys)
    source = FakeSourceStore(data)
    dest = FakeDestStore(metadata_keys, failing_key="sample/c/1")

    with pytest.raises(RuntimeError, match="set failed"):
        asyncio.run(copy_store_async(source, dest, io_concurrency=2))


@pytest.mark.parametrize("io_concurrency", [1, 4])
def test_copy_store_wrapper_produces_identical_contents(io_concurrency):
    source = zarr.storage.MemoryStore()
    dest = zarr.storage.MemoryStore()

    root = zarr.create_group(store=source)
    root.attrs["dataset"] = "example"
    root.create_array(
        "variant_position",
        data=np.array([1, 2, 3], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        "call_genotype",
        data=np.array([[[0, 1]], [[1, 0]], [[0, 0]]], dtype=np.int8),
        chunks=(2, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    copy_store(source, dest, io_concurrency=io_concurrency)

    copied = zarr.open_group(store=dest, mode="r")
    np.testing.assert_array_equal(
        copied["variant_position"][:], root["variant_position"][:]
    )
    np.testing.assert_array_equal(copied["call_genotype"][:], root["call_genotype"][:])
    assert copied.attrs.asdict() == root.attrs.asdict()
