import os
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

import tqdm
from aiostream import stream
from bio2zarr.vcf_utils import ceildiv
from vcztools.constants import FLOAT32_MISSING, INT_MISSING, STR_MISSING
from vcztools.utils import array_dims, make_icechunk_storage
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage._common import make_store

ZARR_METADATA_FILENAMES = frozenset(
    ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")
)


def parse_size(s: str) -> int:
    """Parse a human-readable size string (e.g. '100MB', '2GiB') to bytes."""
    suffixes = [
        ("TiB", 1024**4),
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
        ("TB", 1000**4),
        ("GB", 1000**3),
        ("MB", 1000**2),
        ("KB", 1000),
        ("B", 1),
    ]
    upper = s.strip().upper()
    for suffix, factor in suffixes:
        if upper.endswith(suffix):
            num = s[: len(s) - len(suffix)].strip()
            try:
                return int(float(num) * factor)
            except ValueError as e:
                raise ValueError(f"Cannot parse size: {s!r}") from e
    try:
        return int(s.strip())
    except ValueError as e:
        raise ValueError(f"Cannot parse size: {s!r}") from e


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


def has_variants_axis(arr) -> bool:
    """Whether ``arr``'s first dimension is the variants axis."""
    dims = array_dims(arr)
    return dims is not None and len(dims) > 0 and dims[0] == "variants"


def compute_min_variants_chunk_size(root) -> int:
    """Compute the minimum variants-axis chunk size in a VCZ root.

    By spec ``call_*`` fields define the floor; every variant-only
    field must use a chunk size that is a positive integer multiple of
    it. Two ``call_*`` fields with different chunk sizes are a writer
    bug and raise ``ValueError`` here. When no ``call_*`` field is
    present, falls back to the minimum chunk size across variant-axis
    fields.
    """
    call_sizes: dict[str, int] = {}
    other_sizes: list[int] = []
    for name in root.array_keys():
        arr = root[name]
        if not has_variants_axis(arr):
            continue
        chunk_size = int(arr.chunks[0])
        if name.startswith("call_"):
            call_sizes[name] = chunk_size
        else:
            other_sizes.append(chunk_size)
    if len(call_sizes) > 0:
        sizes_set = set(call_sizes.values())
        if len(sizes_set) > 1:
            raise ValueError(
                f"call_* fields must share a single variants chunk size; "
                f"found {call_sizes}"
            )
        return next(iter(sizes_set))
    if len(other_sizes) > 0:
        return min(other_sizes)
    raise ValueError("no variant-axis fields in store")


def variant_chunk_slices(root, variant_chunks_in_batch=1):
    """A generator returning chunk slices along the variants dimension.

    Batches are in multiples of the minimum variants-axis chunk size as
    found by `compute_min_variants_chunk_size`.
    """
    pos = root["variant_position"]
    size = pos.shape[0]
    min_chunk_size = compute_min_variants_chunk_size(root)
    v_chunksize = min_chunk_size * variant_chunks_in_batch
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


async def copy_store_async(source, dest, *, array_keys=None, io_concurrency):
    source_keys = [key async for key in source.list()]

    if array_keys is not None:
        source_keys = [
            source_key
            for source_key in source_keys
            if source_key.split("/")[0] in array_keys
        ]

    metadata_keys, data_keys = split_metadata_and_data_keys(source_keys)
    prototype = default_buffer_prototype()

    async def _copy_one_key(key):
        buffer = await source.get(key, prototype=prototype)
        if buffer is None:
            raise FileNotFoundError(key)
        await dest.set(key, buffer)

    for key in metadata_keys:
        await _copy_one_key(key)

    if len(data_keys) > 0:
        await stream.map(
            stream.iterate(data_keys), _copy_one_key, task_limit=io_concurrency
        )


def copy_store(source, dest, *, array_keys=None, io_concurrency=None):
    if io_concurrency is None:
        io_concurrency = (os.cpu_count() or 1) * 4
    if io_concurrency < 1:
        raise ValueError("io_concurrency must be greater than or equal to 1")
    source = sync(make_store(source))
    dest = sync(make_store(dest))
    sync(
        copy_store_async(
            source, dest, array_keys=array_keys, io_concurrency=io_concurrency
        )
    )


def copy_store_to_icechunk(source, dest, *, io_concurrency=None):
    """Copy a Zarr store to a new Icechunk store."""
    with icechunk_transaction(dest, "main", create_repo=True, message="create") as dest:
        copy_store(source, dest, io_concurrency=io_concurrency)


@contextmanager
def transaction(
    file_or_url,
    *,
    backend_storage=None,
    branch="main",
    create_repo=False,
    message="update",
):
    """Create a transaction context manager.

    If `backend_storage` is `"icechunk"` the context manager will be an Icechunk
    context manager, otherwise it will be a null context manager that does nothing.
    """
    if backend_storage == "icechunk":
        cm = icechunk_transaction(
            file_or_url, branch, create_repo=create_repo, message=message
        )
    else:
        cm = nullcontext(file_or_url)
    with cm as store:
        yield store


@contextmanager
def icechunk_transaction(file_or_url, branch, *, create_repo=False, message="update"):
    """Open an Icechunk store in a transaction, then commit on completion.

    If `create_repo` is False then the previous commit will be amended so the
    Icechunk repositoy doesn't retain history.
    """
    from icechunk import Repository

    icechunk_storage = make_icechunk_storage(file_or_url)
    if create_repo:
        repo = Repository.create(icechunk_storage)
        with repo.transaction(branch, message=message) as store:
            yield store
    else:
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
