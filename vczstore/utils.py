from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

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


def merge_with(
    l1: list,
    l2: list,
    *,
    equiv: Callable[[Any, Any], bool],
    combine: Callable[[Any, Any], Any],
) -> list:
    """Merge two lists using a pairwise equivalence predicate and combine function.

    Like a key-based list merge, but instead of a key function, uses:
    - equiv(a, b): True if a (from l1) and b (from l2) should be treated as the same
      element and combined. Does not need to be an equivalence relation.
    - combine(a, b): produces the merged element when equiv is True.

    Matching is greedy: each element of l1 is matched to the first unmatched element
    of l2 that satisfies equiv. Unmatched elements pass through unchanged.
    No within-list duplicate checking is performed.

    Raises ValueError if ordering constraints from l1 and l2 conflict.
    """
    # Step 1: greedy matching — l1 drives, first-match wins
    matched_l1: dict[int, int] = {}  # l1 index -> l2 index
    matched_l2: dict[int, int] = {}  # l2 index -> l1 index
    available = list(range(len(l2)))
    for i, a in enumerate(l1):
        for pos, j in enumerate(available):
            if equiv(a, l2[j]):
                matched_l1[i] = j
                matched_l2[j] = i
                available.pop(pos)
                break

    # Step 2: build nodes in first-appearance order (l1 first, then unmatched l2)
    node_items: list[Any] = []
    l1_to_node: dict[int, int] = {}
    l2_to_node: dict[int, int] = {}

    for i, a in enumerate(l1):
        if i in matched_l1:
            j = matched_l1[i]
            idx = len(node_items)
            node_items.append(combine(a, l2[j]))
            l1_to_node[i] = idx
            l2_to_node[j] = idx
        else:
            idx = len(node_items)
            node_items.append(a)
            l1_to_node[i] = idx

    for j, b in enumerate(l2):
        if j not in matched_l2:
            idx = len(node_items)
            node_items.append(b)
            l2_to_node[j] = idx

    n = len(node_items)
    rank = {i: i for i in range(n)}
    graph: dict[int, set[int]] = {i: set() for i in range(n)}
    in_degree: dict[int, int] = {i: 0 for i in range(n)}

    # Step 3: ordering constraints from both lists
    for i in range(len(l1) - 1):
        a, b = l1_to_node[i], l1_to_node[i + 1]
        if a != b and b not in graph[a]:
            graph[a].add(b)
            in_degree[b] += 1

    for j in range(len(l2) - 1):
        a, b = l2_to_node[j], l2_to_node[j + 1]
        if a != b and b not in graph[a]:
            graph[a].add(b)
            in_degree[b] += 1

    # Step 4: topological sort, rank as tiebreaker
    queue = sorted([i for i in range(n) if in_degree[i] == 0], key=rank.__getitem__)
    result: list[int] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        newly_free = []
        for s in graph[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                newly_free.append(s)
        queue = sorted(queue + newly_free, key=rank.__getitem__)

    if len(result) < n:
        raise ValueError("Cannot merge lists: ordering conflict detected")

    return [node_items[i] for i in result]
