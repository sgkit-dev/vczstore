import logging
import os
import pathlib
import tempfile
from contextlib import suppress
from itertools import product

import numpy as np
import zarr
from aiostream import stream
from vcztools.utils import array_dims, open_zarr
from zarr.core.sync import sync

from vczstore.normalise import normalise
from vczstore.utils import compute_min_variants_chunk_size, copy_store, transaction

logger = logging.getLogger(__name__)


def _require_normalise(root1, root2):
    n_variants1 = root1["variant_contig"].shape[0]
    n_variants2 = root2["variant_contig"].shape[0]
    if n_variants1 != n_variants2:
        return True
    fields = ["contig_id", "variant_contig", "variant_position"]
    if "normalise_new_alleles" not in root2["variant_allele"].attrs:
        # normalise has not been called
        fields.append("variant_allele")
    for field in fields:
        if not np.array_equal(root1[field][:], root2[field][:]):
            return True
    return False


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
    dst_arr, src_arr, *, dst_num_sample_chunks, io_concurrency
):
    async def copy_chunk(src_coords):
        dst_coords = (
            src_coords[:1] + (dst_num_sample_chunks + src_coords[1],) + src_coords[2:]
        )
        src_key = src_arr.store_path / src_arr.metadata.encode_chunk_key(src_coords)
        dst_key = dst_arr.store_path / dst_arr.metadata.encode_chunk_key(dst_coords)
        buf = await src_key.get()
        if buf is None:
            # Sparse source chunks must clear any stale destination chunk.
            with suppress(FileNotFoundError):
                await dst_key.delete()
        else:
            await dst_key.set(buf)

    await stream.map(
        stream.iterate(product(*[range(n) for n in src_arr.cdata_shape])),
        copy_chunk,
        task_limit=io_concurrency,
    )


def temp_norm_path(prefix=None):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        return pathlib.Path(tmp) / "vcz_norm"


def append(
    vcz1,
    vcz2,
    *,
    haploid_contigs=None,
    allow_new_alleles=False,
    variant_chunks_in_batch=None,
    show_progress=False,
    backend_storage=None,
    io_concurrency=None,
    require_direct_copy=False,
):
    """Append vcz2 to vcz1 in place"""
    if io_concurrency is None:
        io_concurrency = (os.cpu_count() or 1) * 4
    if io_concurrency < 1:
        raise ValueError("io_concurrency must be greater than or equal to 1")

    with transaction(vcz1, backend_storage=backend_storage, message="append") as vcz1:
        root1 = open_zarr(vcz1, mode="r+", backend_storage=backend_storage)
        root2 = zarr.open(vcz2, mode="r")  # assume local

        # check preconditions
        sample_id1 = root1["sample_id"]
        sample_id2 = root2["sample_id"]
        common_samples = np.intersect1d(sample_id1, sample_id2)
        if common_samples.shape[0] > 0:
            raise ValueError(f"Duplicate samples found: {common_samples}")

        if _require_normalise(root1, root2):
            # TODO: use a context manager to delete temp path after use
            vcz2_norm = temp_norm_path(prefix="vczstore")
            normalise(
                vcz1,
                vcz2,
                vcz2_norm,
                haploid_contigs=haploid_contigs,
                allow_new_alleles=allow_new_alleles,
                variant_chunks_in_batch=variant_chunks_in_batch,
                show_progress=show_progress,
                backend_storage=backend_storage,
            )
            vcz2 = vcz2_norm
            root2 = zarr.open(vcz2, mode="r")  # assume local

        min_chunk_size1 = compute_min_variants_chunk_size(root1)
        min_chunk_size2 = compute_min_variants_chunk_size(root2)
        if min_chunk_size1 != min_chunk_size2:
            raise ValueError(
                f"append requires stores have matching minimum variants chunk sizes. "
                f"First has {min_chunk_size1}, second has {min_chunk_size2}"
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
                arr1_num_sample_chunks = arr1.cdata_shape[1]
                _assert_append_arrays_compatible(var, arr1, arr2)
                call_arrays.append((var, arr1, arr2, arr1_num_sample_chunks))

        # append samples
        old_num_samples = sample_id1.shape[0]
        incoming_num_samples = sample_id2.shape[0]
        new_num_samples = old_num_samples + incoming_num_samples
        logger.info(
            f"Old num samples: {old_num_samples}, "
            f"incoming num samples: {incoming_num_samples}, "
            f"new num samples: {new_num_samples}"
        )

        if require_direct_copy:
            for name, arr1, arr2, _ in call_arrays:
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
        for _, arr, _, _ in call_arrays:
            arr.resize((arr.shape[0], new_num_samples, *arr.shape[2:]))

        sample_id1[old_num_samples:new_num_samples] = sample_id2[:]

        with zarr.config.set({"async.concurrency": io_concurrency}):
            for name, arr1, arr2, arr1_num_sample_chunks in call_arrays:
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
                    logger.debug(f"Copying encoded chunks for {name} (fast path)")
                    sync(
                        _copy_encoded_chunks(
                            arr1,
                            arr2,
                            dst_num_sample_chunks=arr1_num_sample_chunks,
                            io_concurrency=io_concurrency,
                        )
                    )

                if direct_count < incoming_num_samples:
                    logger.debug(f"Copying data for {name} (not fast path)")
                    arr1[:, old_num_samples + direct_count : new_num_samples, ...] = (
                        arr2[:, direct_count:incoming_num_samples, ...]
                    )

        # normalise will set the 'normalise_new_alleles' flag if there are new alleles
        normalise_new_alleles = root2["variant_allele"].attrs.get(
            "normalise_new_alleles", False
        )

        if normalise_new_alleles:
            logger.info("Overwriting variant_allele array since new alleles present")
            copy_store(vcz2, vcz1, array_keys=["variant_allele"])
