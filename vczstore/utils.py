from collections.abc import Callable
from typing import Any

import tqdm
from vcztools.constants import FLOAT32_MISSING, INT_MISSING, STR_MISSING
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


def variant_chunk_slices(root):
    """A generator returning chunk slices along the variants dimension."""
    pos = root["variant_position"]
    size = pos.shape[0]
    v_chunksize = pos.chunks[0]
    num_chunks = pos.cdata_shape[0]
    for v_chunk in range(num_chunks):
        start = v_chunksize * v_chunk
        end = min(v_chunksize * (v_chunk + 1), size)
        yield slice(start, end)


def variants_progress(n_variants, title, show_progress=False):
    return tqdm.tqdm(
        total=n_variants,
        desc=f"{title:>8}",
        unit_scale=True,
        unit="vars",
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


def merge_lists(l1: list, l2: list, *, key: Callable[[Any], Any] = lambda x: x) -> list:
    """Merge two lists preserving the relative order of elements from both inputs.

    Returns a list containing all elements from l1 and l2, deduplicated by key,
    ordered consistently with both inputs. Original items (not key values) are returned;
    when the same key appears in both lists, the item from l1 is kept.

    Args:
        l1: First ordered list.
        l2: Second ordered list.
        key: Function extracting a hashable identity from each element. Defaults to
             the element itself. Two elements with the same key are considered equal.

    Raises ValueError if the ordering constraints from l1 and l2 conflict,
    or if either input list contains duplicate elements (by key).
    """
    for lst, name in ((l1, "l1"), (l2, "l2")):
        keys = [key(item) for item in lst]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Input {name} contains duplicate elements")

    # Assign a stable rank to each element keyed by key(item).
    # When the same key appears in both lists, the item from l1 is kept.
    rank: dict[Any, int] = {}
    items: dict[Any, Any] = {}  # key -> original item
    for item in l1 + l2:
        k = key(item)
        if k not in rank:
            rank[k] = len(rank)
            items[k] = item

    all_keys = list(rank.keys())
    graph: dict[Any, set] = {k: set() for k in all_keys}
    in_degree: dict[Any, int] = {k: 0 for k in all_keys}

    for lst in (l1, l2):
        for a, b in zip(lst, lst[1:]):
            ka, kb = key(a), key(b)
            if kb not in graph[ka]:
                graph[ka].add(kb)
                in_degree[kb] += 1

    queue = sorted([k for k in all_keys if in_degree[k] == 0], key=rank.__getitem__)
    result_keys: list = []

    while queue:
        k = queue.pop(0)
        result_keys.append(k)
        newly_free = []
        for successor in graph[k]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                newly_free.append(successor)
        queue = sorted(queue + newly_free, key=rank.__getitem__)

    if len(result_keys) < len(all_keys):
        remaining = [items[k] for k in all_keys if k not in set(result_keys)]
        raise ValueError(
            f"Cannot merge lists: ordering conflict detected among elements {remaining}"
        )
    return [items[k] for k in result_keys]
