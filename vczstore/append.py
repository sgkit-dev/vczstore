import logging

import numpy as np
from vcztools.utils import array_dims
from zarr.api import asynchronous as zarr_async
from zarr.core.sync import sync

from vczstore.concurrency import resolve_io_concurrency, run_bounded
from vczstore.utils import variant_chunk_slices_for_array, variants_progress

logger = logging.getLogger(__name__)


def append(vcz1, vcz2, *, show_progress=False, io_concurrency=None):
    """Append vcz2 to vcz1 in place."""
    resolved_io_concurrency = resolve_io_concurrency(io_concurrency)
    sync(
        _append_async(
            vcz1,
            vcz2,
            show_progress=show_progress,
            io_concurrency=resolved_io_concurrency,
        )
    )


def _sample_range_selection(array, variant_selection, sample_selection):
    return (variant_selection, sample_selection) + (slice(None),) * (array.ndim - 2)


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


async def _append_async(vcz1, vcz2, *, show_progress=False, io_concurrency):
    root1, root2 = await zarr_async.open_group(
        store=vcz1, mode="r+"
    ), await zarr_async.open_group(store=vcz2, mode="r")

    variant_contig1 = await root1.getitem("variant_contig")
    variant_contig2 = await root2.getitem("variant_contig")
    n_variants1 = variant_contig1.shape[0]
    n_variants2 = variant_contig2.shape[0]
    if n_variants1 != n_variants2:
        raise ValueError(
            "Stores being appended must have same number of variants. "
            f"First has {n_variants1}, second has {n_variants2}"
        )

    for field in ("contig_id", "variant_contig", "variant_position", "variant_allele"):
        values1 = await (await root1.getitem(field)).getitem(slice(None))
        values2 = await (await root2.getitem(field)).getitem(slice(None))
        if np.any(values1 != values2):
            raise ValueError(
                f"Stores being appended must have same values for field '{field}'"
            )

    sample_id1 = await root1.getitem("sample_id")
    sample_id2 = await root2.getitem("sample_id")
    sample_id2_values = await sample_id2.getitem(slice(None))

    root2_arrays = {name: arr async for name, arr in root2.arrays()}
    call_arrays = []
    async for name, arr1 in root1.arrays():
        if name.startswith("call_"):
            call_arrays.append((name, arr1, root2_arrays[name]))

    _assert_variant_chunk_alignment(
        [(name, arr1) for name, arr1, _ in call_arrays],
        variant_chunk_size=variant_contig1.chunks[0],
        operation="append",
    )

    old_num_samples = sample_id1.shape[0]
    new_num_samples = old_num_samples + sample_id2.shape[0]
    for _, arr1, _ in call_arrays:
        if arr1.ndim == 2:
            continue
        elif arr1.ndim == 3:
            continue
        else:
            raise ValueError("unsupported number of array_dims")

    await sample_id1.resize((new_num_samples,))
    await sample_id1.setitem(slice(old_num_samples, new_num_samples), sample_id2_values)

    for _, arr1, _ in call_arrays:
        if arr1.ndim == 2:
            await arr1.resize((arr1.shape[0], new_num_samples))
        else:
            await arr1.resize((arr1.shape[0], new_num_samples, arr1.shape[2]))

    variant_slices = list(variant_chunk_slices_for_array(variant_contig1))
    sample_selection = slice(old_num_samples, new_num_samples)

    async def worker(v_sel):
        for _, arr1, arr2 in call_arrays:
            data = await arr2.getitem(v_sel)
            await arr1.setitem(
                _sample_range_selection(arr1, v_sel, sample_selection), data
            )

    with variants_progress(n_variants1, "Append", show_progress) as pbar:
        await run_bounded(
            variant_slices,
            worker,
            max_concurrency=io_concurrency,
            on_item_done=lambda v_sel: pbar.update(v_sel.stop - v_sel.start),
        )
