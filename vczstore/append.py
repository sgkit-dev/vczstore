import asyncio
import logging
import os
from contextlib import suppress
from itertools import product

import numpy as np
import zarr
from vcztools.utils import array_dims
from zarr.core.sync import sync

logger = logging.getLogger(__name__)


def _assert_variant_chunk_alignment(arrays, *, variant_chunk_size, operation):
    for name, arr in arrays:
        dims = array_dims(arr)
        if dims is None or len(dims) == 0 or dims[0] != "variants":
            raise ValueError(f"{operation} requires {name!r} to be chunked on variants")
        if arr.chunks is None or arr.chunks[0] != variant_chunk_size:
            raise ValueError(
                f"{operation} requires {name!r} to use VCZ-aligned variant chunks of "
                f"size {variant_chunk_size}"
            )


def _assert_append_arrays_compatible(name, arr1, arr2):
    dims1 = array_dims(arr1)
    dims2 = array_dims(arr2)
    if (
        dims1 is None
        or len(dims1) < 2
        or tuple(dims1[:2])
        != (
            "variants",
            "samples",
        )
    ):
        raise ValueError(f"append requires {name!r} to use variants/samples dimensions")
    if dims1 != dims2:
        raise ValueError(
            f"append requires {name!r} to have matching dimensions. "
            f"First has {dims1}, second has {dims2}"
        )
    if arr1.ndim != arr2.ndim:
        raise ValueError(
            f"append requires {name!r} to have matching number of dimensions. "
            f"First has {arr1.ndim}, second has {arr2.ndim}"
        )
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError(
            f"append requires {name!r} to have matching number of variants. "
            f"First has {arr1.shape[0]}, second has {arr2.shape[0]}"
        )
    if arr1.ndim > 2 and arr1.shape[2:] != arr2.shape[2:]:
        raise ValueError(
            f"append requires {name!r} to have matching trailing dimensions. "
            f"First has {arr1.shape[2:]}, second has {arr2.shape[2:]}"
        )


def _copy_encoded_chunks_error(name, arr1, arr2):
    if arr1.chunks != arr2.chunks:
        return (
            "direct append requires matching chunks for encoded chunk copy. "
            f"{name!r} has chunks {arr1.chunks} and {arr2.chunks}"
        )
    # Shape changes during append and attributes do not affect encoded chunk bytes.
    if {**arr1.metadata.to_dict(), "shape": None, "attributes": None} != {
        **arr2.metadata.to_dict(),
        "shape": None,
        "attributes": None,
    }:
        return (
            "direct append requires matching encoded chunk metadata. "
            f"{name!r} cannot be copied chunk-by-chunk"
        )
    return None


async def _copy_encoded_chunks(
    dst_arr, src_arr, *, src_start, dst_start, count, io_concurrency
):
    async def copy_chunk(src_coords, dst_coords):
        src_key = src_arr.store_path / src_arr.metadata.encode_chunk_key(src_coords)
        dst_key = dst_arr.store_path / dst_arr.metadata.encode_chunk_key(dst_coords)
        buf = await src_key.get()
        if buf is None:
            # Sparse source chunks must clear any stale destination chunk.
            with suppress(FileNotFoundError):
                await dst_key.delete()
        else:
            await dst_key.set(buf)

    async def wait_for_one(tasks):
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        return tasks

    tasks = set()
    try:
        for variant_chunk in range(src_arr.cdata_shape[0]):
            for sample_offset in range(count // dst_arr.chunks[1]):
                for extra_chunk_coords in product(
                    *[range(n) for n in src_arr.cdata_shape[2:]]
                ):
                    src_coords = (
                        variant_chunk,
                        src_start // src_arr.chunks[1] + sample_offset,
                        *extra_chunk_coords,
                    )
                    dst_coords = (
                        variant_chunk,
                        dst_start // dst_arr.chunks[1] + sample_offset,
                        *extra_chunk_coords,
                    )
                    if len(tasks) >= io_concurrency:
                        tasks = await wait_for_one(tasks)
                    tasks.add(asyncio.create_task(copy_chunk(src_coords, dst_coords)))
        while tasks:
            # Surface errors as soon as they happen
            tasks = await wait_for_one(tasks)
    # Cancel remaining work on error
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def append(vcz1, vcz2, *, io_concurrency=None, require_direct_copy=False):
    """Append vcz2 to vcz1 in place"""
    if io_concurrency is None:
        io_concurrency = (os.cpu_count() or 1) * 4
    if io_concurrency < 1:
        raise ValueError("io_concurrency must be greater than or equal to 1")
    root1 = zarr.open(vcz1, mode="r+")
    root2 = zarr.open(vcz2, mode="r")

    # check preconditions
    n_variants1 = root1["variant_contig"].shape[0]
    n_variants2 = root2["variant_contig"].shape[0]
    if n_variants1 != n_variants2:
        raise ValueError(
            "Stores being appended must have same number of variants. "
            f"First has {n_variants1}, second has {n_variants2}"
        )
    for field in ("contig_id", "variant_contig", "variant_position", "variant_allele"):
        if not np.array_equal(root1[field][:], root2[field][:]):
            raise ValueError(
                f"Stores being appended must have same values for field '{field}'"
            )

    call_arrays = []
    for var in root1.keys():
        if var.startswith("call_"):
            if var not in root2:
                raise ValueError(
                    f"append requires {var!r} to be present in both stores"
                )
            arr1 = root1[var]
            arr2 = root2[var]
            _assert_append_arrays_compatible(var, arr1, arr2)
            call_arrays.append((var, arr1, arr2))
    _assert_variant_chunk_alignment(
        [(var, arr1) for var, arr1, _ in call_arrays],
        variant_chunk_size=root1["variant_contig"].chunks[0],
        operation="append",
    )
    _assert_variant_chunk_alignment(
        [(var, arr2) for var, _, arr2 in call_arrays],
        variant_chunk_size=root2["variant_contig"].chunks[0],
        operation="append",
    )

    # append samples
    sample_id1 = root1["sample_id"]
    sample_id2 = root2["sample_id"]

    old_num_samples = sample_id1.shape[0]
    incoming_num_samples = sample_id2.shape[0]
    new_num_samples = old_num_samples + incoming_num_samples

    if require_direct_copy:
        for name, arr1, arr2 in call_arrays:
            sample_chunk_size = arr1.chunks[1]
            if (
                old_num_samples % sample_chunk_size
                or incoming_num_samples % sample_chunk_size
            ):
                raise ValueError(
                    "direct-only append requires the destination sample count and "
                    "incoming sample count to be sample chunk-aligned. "
                    f"{name!r} uses sample chunks of size "
                    f"{sample_chunk_size}"
                )
            if error := _copy_encoded_chunks_error(name, arr1, arr2):
                raise ValueError(error)

    sample_id1.resize((new_num_samples,))

    # resize genotype fields
    for _, arr, _ in call_arrays:
        arr.resize((arr.shape[0], new_num_samples, *arr.shape[2:]))

    sample_id1[old_num_samples:new_num_samples] = sample_id2[:]

    with zarr.config.set({"async.concurrency": io_concurrency}):
        for name, arr1, arr2 in call_arrays:
            sample_chunk_size = arr1.chunks[1]
            if (
                old_num_samples % sample_chunk_size == 0
                and _copy_encoded_chunks_error(name, arr1, arr2) is None
            ):
                direct_count = (
                    incoming_num_samples // sample_chunk_size
                ) * sample_chunk_size
            else:
                direct_count = 0

            if direct_count:
                sync(
                    _copy_encoded_chunks(
                        arr1,
                        arr2,
                        src_start=0,
                        dst_start=old_num_samples,
                        count=direct_count,
                        io_concurrency=io_concurrency,
                    )
                )

            if direct_count < incoming_num_samples:
                arr1[:, old_num_samples + direct_count : new_num_samples, ...] = arr2[
                    :, direct_count:incoming_num_samples, ...
                ]
