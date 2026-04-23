import logging

import numpy as np
from vcztools.utils import array_dims, search
from zarr.api import asynchronous as zarr_async
from zarr.core.sync import sync

from vczstore.concurrency import resolve_io_concurrency, run_bounded
from vczstore.utils import (
    missing_val,
    variant_chunk_slices_for_array,
    variants_progress,
)

logger = logging.getLogger(__name__)


def remove(vcz, sample_id, *, show_progress=False, io_concurrency=None):
    """Remove a sample from vcz and overwrite with missing data."""
    resolved_io_concurrency = resolve_io_concurrency(io_concurrency)
    sync(
        _remove_async(
            vcz,
            sample_id,
            show_progress=show_progress,
            io_concurrency=resolved_io_concurrency,
        )
    )


def _sample_selection(array, variant_selection, sample_selection):
    return (variant_selection, sample_selection) + (slice(None),) * (array.ndim - 2)


def _assert_variant_chunk_alignment(arrays, *, variant_chunk_size, operation):
    for name, arr in arrays:
        dims = array_dims(arr)
        if (
            dims is None
            or len(dims) < 2
            or dims[0] != "variants"
            or dims[1] != "samples"
        ):
            raise ValueError(
                f"{operation} requires {name!r} to use variants/samples dimensions"
            )
        if arr.chunks is None or arr.chunks[0] != variant_chunk_size:
            raise ValueError(
                f"{operation} requires {name!r} to use VCZ-aligned variant chunks of "
                f"size {variant_chunk_size}"
            )


async def _remove_async(vcz, sample_id, *, show_progress=False, io_concurrency):
    root = await zarr_async.open_group(store=vcz, mode="r+")
    variant_contig = await root.getitem("variant_contig")
    n_variants = variant_contig.shape[0]
    sample_id_arr = await root.getitem("sample_id")
    all_samples = await sample_id_arr.getitem(slice(None))

    unknown_samples = np.setdiff1d(sample_id, all_samples)
    if len(unknown_samples) > 0:
        raise ValueError(f"unrecognised sample: {sample_id}")
    sample_selection = int(np.asarray(search(all_samples, sample_id)).item())

    target_arrays = []
    async for name, arr in root.arrays():
        dims = array_dims(arr)
        if (
            name.startswith("call_")
            and dims is not None
            and len(dims) >= 2
            and dims[0] == "variants"
            and dims[1] == "samples"
        ):
            target_arrays.append((name, arr))

    _assert_variant_chunk_alignment(
        target_arrays,
        variant_chunk_size=variant_contig.chunks[0],
        operation="remove",
    )

    await sample_id_arr.setitem(sample_selection, "")

    variant_slices = list(variant_chunk_slices_for_array(variant_contig))

    async def worker(v_sel):
        for _, arr in target_arrays:
            await arr.setitem(
                _sample_selection(arr, v_sel, sample_selection), missing_val(arr)
            )

    with variants_progress(n_variants, "Remove", show_progress) as pbar:
        await run_bounded(
            variant_slices,
            worker,
            max_concurrency=io_concurrency,
            on_item_done=lambda v_sel: pbar.update(v_sel.stop - v_sel.start),
        )
