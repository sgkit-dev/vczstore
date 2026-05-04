import logging
from dataclasses import dataclass, field

import numpy as np
import zarr
from bio2zarr.vcz import VcfZarrIndexer
from bio2zarr.zarr_utils import (
    STRING_DTYPE_NAME,
    create_empty_group_array,
    create_group_array,
    get_compressor_config,
)
from vcztools.constants import STR_FILL, STR_MISSING
from vcztools.utils import array_dims, open_zarr

from vczstore.utils import copy_store, merge_with, progress_bar

logger = logging.getLogger(__name__)


def _strip_padding(arr: np.ndarray) -> np.ndarray:
    return arr[(arr != STR_MISSING) & (arr != STR_FILL)]


def _can_merge_variants(allele_a: np.ndarray, allele_b: np.ndarray) -> bool:
    """Return True if two VCZ variant rows can be merged under bcftools -m none."""
    ref_a, ref_b = allele_a[0], allele_b[0]
    if ref_a != ref_b:
        return False
    alt_a = _strip_padding(allele_a[1:]).tolist()
    alt_b = _strip_padding(allele_b[1:]).tolist()
    # len == 0 after stripping means ref-only (ALT was "." = STR_MISSING)
    if len(alt_a) == 0 or len(alt_b) == 0:
        return True
    return bool(set(alt_a) & set(alt_b))


def _merge_alts(alts1: list[str], alts2: list[str]) -> list[str]:
    """Return merged ALT list in first-occurrence order, like merge_alleles."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for alts in (alts1, alts2):
        if alts == ["."]:
            continue
        for a in alts:
            if a not in seen_set:
                seen.append(a)
                seen_set.add(a)
    return seen if seen else ["."]


def _merge_ids(id1: str, id2: str) -> str:
    """Join two variant IDs with ";", dropping "." (missing) entries and
    deduplicating identical IDs."""
    parts = [x for x in (id1, id2) if x != "."]
    unique_parts = list(dict.fromkeys(parts))
    return ";".join(unique_parts) if unique_parts else "."


def _merge_filters(f1: np.ndarray | None, f2: np.ndarray | None) -> np.ndarray | None:
    """Return the union (OR) of two filter bool arrays."""
    if f1 is None and f2 is None:
        return None
    if f1 is None:
        return f2.copy() if f2 is not None else None
    if f2 is None:
        return f1.copy()
    return f1 | f2


@dataclass
class _VCZRecord:
    contig: int
    position: int
    alleles: list[str]  # [ref, alt1, ...], no padding
    length: int = 0
    id: str = "."
    quality: float = float("nan")
    filter_: np.ndarray | None = field(default=None, repr=False)


def _combine_records(a: _VCZRecord, b: _VCZRecord) -> _VCZRecord:
    merged_alts = _merge_alts(a.alleles[1:], b.alleles[1:])
    return _VCZRecord(
        contig=a.contig,
        position=a.position,
        alleles=[a.alleles[0]] + merged_alts,
        length=max(a.length, b.length),
        id=_merge_ids(a.id, b.id),
        quality=float(np.fmax(a.quality, b.quality)),
        filter_=_merge_filters(a.filter_, b.filter_),
    )


def _record_from_row(
    allele_row,
    contig,
    position,
    length=0,
    id=".",
    quality=float("nan"),
    filter_=None,
) -> _VCZRecord:
    alleles = [str(a) for a in _strip_padding(allele_row).tolist()] or ["."]
    return _VCZRecord(
        alleles=alleles,
        contig=int(contig),
        position=int(position),
        length=int(length),
        id=str(id),
        quality=float(quality),
        filter_=filter_,
    )


def _compute_merged_variants(
    root1, root2, show_progress=False
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
    """Compute merged variant arrays.

    Uses vectorized numpy for the common cases (vcz1-only, vcz2-only, and exact
    allele matches at simple sites), falling back to a Python loop only for
    complex shared positions (>1 variant per position in either store) and
    simple shared sites whose alleles are not byte-identical.

    Returns (contig, position, length, id, quality, filter_, allele).
    Optional fields (length, id, quality, filter_) are None when either store
    lacks the corresponding arrays. filter_ requires identical filter_id arrays.
    """
    with progress_bar(6, "Read", show_progress, unit="array") as pbar:
        vc1 = root1["variant_contig"][:]
        pbar.update()
        vp1 = root1["variant_position"][:]
        pbar.update()
        va1 = root1["variant_allele"][:]
        pbar.update()
        vc2 = root2["variant_contig"][:]
        pbar.update()
        vp2 = root2["variant_position"][:]
        pbar.update()
        va2 = root2["variant_allele"][:]
        pbar.update()

    has_length = "variant_length" in root1 and "variant_length" in root2
    if has_length:
        vl1 = root1["variant_length"][:]
        vl2 = root2["variant_length"][:]

    has_id = "variant_id" in root1 and "variant_id" in root2
    if has_id:
        vi1 = root1["variant_id"][:]
        vi2 = root2["variant_id"][:]

    has_quality = "variant_quality" in root1 and "variant_quality" in root2
    if has_quality:
        vq1 = root1["variant_quality"][:]
        vq2 = root2["variant_quality"][:]

    has_filter = (
        "variant_filter" in root1
        and "variant_filter" in root2
        and "filter_id" in root1
        and "filter_id" in root2
    )
    if has_filter:
        fi1 = root1["filter_id"][:]
        fi2 = root2["filter_id"][:]
        if fi1.shape != fi2.shape or not np.all(fi1 == fi2):
            raise ValueError("filter_id fields must be identical")
        vf1 = root1["variant_filter"][:]
        vf2 = root2["variant_filter"][:]

    n1, n2 = len(vc1), len(vc2)

    if n1 == 0 and n2 == 0:
        empty_int = np.array([], dtype=np.int32)
        return (
            empty_int,
            empty_int,
            None,
            None,
            None,
            None,
            np.empty((0, 1), dtype=STRING_DTYPE_NAME),
        )

    # Encode (contig, position) as int64 keys (same approach as index_variants)
    pos_stride = int(max(vp1.max() if n1 > 0 else 0, vp2.max() if n2 > 0 else 0)) + 1
    max_contig = int(max(vc1.max() if n1 > 0 else 0, vc2.max() if n2 > 0 else 0))
    if max_contig * pos_stride > np.iinfo(np.int64).max:
        raise ValueError(
            f"Cannot encode variants as int64 keys: "
            f"max_contig={max_contig} * pos_stride={pos_stride} overflows int64"
        )
    keys1 = vc1.astype(np.int64) * pos_stride + vp1.astype(np.int64)
    keys2 = vc2.astype(np.int64) * pos_stride + vp2.astype(np.int64)

    # For each vcz1 variant: group sizes in vcz2 and within vcz1 at same position
    l2_for_1 = np.searchsorted(keys2, keys1, side="left")
    r2_for_1 = np.searchsorted(keys2, keys1, side="right")
    n2_at_1 = r2_for_1 - l2_for_1

    l1_self = np.searchsorted(keys1, keys1, side="left")
    r1_self = np.searchsorted(keys1, keys1, side="right")
    n1_at_1 = r1_self - l1_self

    # For each vcz2 variant: group sizes in vcz1 and within vcz2 at same position
    l1_for_2 = np.searchsorted(keys1, keys2, side="left")
    r1_for_2 = np.searchsorted(keys1, keys2, side="right")
    n1_at_2 = r1_for_2 - l1_for_2

    l2_self = np.searchsorted(keys2, keys2, side="left")
    r2_self = np.searchsorted(keys2, keys2, side="right")
    n2_at_2 = r2_self - l2_self

    # Classify vcz1 variants
    vcz1_only = n2_at_1 == 0  # no vcz2 at this position → pass through as-is
    vcz1_simple_shared = ~vcz1_only & (n1_at_1 == 1) & (n2_at_1 == 1)
    vcz1_complex_shared = ~vcz1_only & ~vcz1_simple_shared

    # Classify vcz2 variants
    vcz2_only = n1_at_2 == 0  # no vcz1 at this position → pass through as-is
    vcz2_complex_shared = ~vcz2_only & ((n1_at_2 > 1) | (n2_at_2 > 1))

    # --- Simple shared sites: 1 variant per position in each store ---
    simple_i1 = np.where(vcz1_simple_shared)[0]
    simple_i2 = l2_for_1[vcz1_simple_shared]  # corresponding vcz2 index (exactly 1)

    # Exact allele match (handles different widths between stores)
    if len(simple_i1) > 0:
        min_w = min(va1.shape[1], va2.shape[1])
        common_equal = np.all(va1[simple_i1, :min_w] == va2[simple_i2, :min_w], axis=1)
        if va1.shape[1] > va2.shape[1]:
            extra_fill = np.all(va1[simple_i1, va2.shape[1] :] == STR_FILL, axis=1)
        elif va2.shape[1] > va1.shape[1]:
            extra_fill = np.all(va2[simple_i2, va1.shape[1] :] == STR_FILL, axis=1)
        else:
            extra_fill = np.ones(len(simple_i1), dtype=bool)
        exact = common_equal & extra_fill
    else:
        exact = np.zeros(0, dtype=bool)

    exact_i1 = simple_i1[exact]
    exact_i2 = simple_i2[exact]
    non_exact_i1 = simple_i1[~exact]
    non_exact_i2 = simple_i2[~exact]

    # --- Complex shared sites: Python loop via merge_with ---
    complex_keys_1 = keys1[vcz1_complex_shared]
    complex_keys_2 = keys2[vcz2_complex_shared]
    if len(complex_keys_1) + len(complex_keys_2) > 0:
        complex_keys = np.unique(np.concatenate([complex_keys_1, complex_keys_2]))
    else:
        complex_keys = np.array([], dtype=np.int64)

    # Python fallback: each entry is
    # (sort_key, tiebreak, contig, position, alleles, length, id, quality, filter_arr)
    # tiebreak=0 for vcz1-origin, 1 for vcz2-origin; used to order within same position
    python_output: list[
        tuple[int, int, int, int, list[str], int, str, float, np.ndarray | None]
    ] = []

    for i1, i2 in zip(non_exact_i1, non_exact_i2):
        a1, a2 = va1[i1], va2[i2]
        l1_val = int(vl1[i1]) if has_length else 0
        l2_val = int(vl2[i2]) if has_length else 0
        id1_val = str(vi1[i1]) if has_id else "."
        id2_val = str(vi2[i2]) if has_id else "."
        q1_val = float(vq1[i1]) if has_quality else float("nan")
        q2_val = float(vq2[i2]) if has_quality else float("nan")
        f1_val = vf1[i1].copy() if has_filter else None
        f2_val = vf2[i2].copy() if has_filter else None
        if _can_merge_variants(a1, a2):
            alt1 = _strip_padding(a1[1:]).tolist()
            alt2 = _strip_padding(a2[1:]).tolist()
            alleles = [str(a1[0])] + _merge_alts(alt1, alt2)
            python_output.append(
                (
                    int(keys1[i1]),
                    0,
                    int(vc1[i1]),
                    int(vp1[i1]),
                    alleles,
                    max(l1_val, l2_val),
                    _merge_ids(id1_val, id2_val),
                    float(np.fmax(q1_val, q2_val)),
                    _merge_filters(f1_val, f2_val),
                )
            )
        else:
            alleles1 = [str(a) for a in _strip_padding(a1)] or ["."]
            alleles2 = [str(a) for a in _strip_padding(a2)] or ["."]
            python_output.append(
                (
                    int(keys1[i1]),
                    0,
                    int(vc1[i1]),
                    int(vp1[i1]),
                    alleles1,
                    l1_val,
                    id1_val,
                    q1_val,
                    f1_val,
                )
            )
            python_output.append(
                (
                    int(keys2[i2]),
                    1,
                    int(vc2[i2]),
                    int(vp2[i2]),
                    alleles2,
                    l2_val,
                    id2_val,
                    q2_val,
                    f2_val,
                )
            )

    for key in complex_keys:
        i1s = int(np.searchsorted(keys1, key, side="left"))
        i1e = int(np.searchsorted(keys1, key, side="right"))
        i2s = int(np.searchsorted(keys2, key, side="left"))
        i2e = int(np.searchsorted(keys2, key, side="right"))
        g1 = [
            _record_from_row(
                va1[i],
                vc1[i],
                vp1[i],
                vl1[i] if has_length else 0,
                str(vi1[i]) if has_id else ".",
                float(vq1[i]) if has_quality else float("nan"),
                vf1[i].copy() if has_filter else None,
            )
            for i in range(i1s, i1e)
        ]
        g2 = [
            _record_from_row(
                va2[i],
                vc2[i],
                vp2[i],
                vl2[i] if has_length else 0,
                str(vi2[i]) if has_id else ".",
                float(vq2[i]) if has_quality else float("nan"),
                vf2[i].copy() if has_filter else None,
            )
            for i in range(i2s, i2e)
        ]
        for rank, rec in enumerate(
            merge_with(
                g1,
                g2,
                equiv=lambda a, b: _can_merge_variants(
                    np.array(a.alleles), np.array(b.alleles)
                ),
                combine=_combine_records,
            )
        ):
            python_output.append(
                (
                    int(key),
                    rank,
                    rec.contig,
                    rec.position,
                    rec.alleles,
                    rec.length,
                    rec.id,
                    rec.quality,
                    rec.filter_,
                )
            )

    # --- Assemble output arrays ---
    # Each record carries a (sort_key, tiebreak) pair for final ordering.
    # vcz1-origin: tiebreak=0; vcz2-origin: tiebreak=1.
    # Records at the same key from different categories cannot both be vcz1/vcz2-only,
    # so the ordering is unambiguous.

    vcz1_only_idx = np.where(vcz1_only)[0]
    vcz2_only_idx = np.where(vcz2_only)[0]
    n_v1 = len(vcz1_only_idx)
    n_v2 = len(vcz2_only_idx)
    n_ex = len(exact_i1)
    n_py = len(python_output)
    n_out = n_v1 + n_v2 + n_ex + n_py

    # Sort keys and tiebreaks for all output records
    all_keys = np.empty(n_out, dtype=np.int64)
    all_keys[:n_v1] = keys1[vcz1_only_idx]
    all_keys[n_v1 : n_v1 + n_v2] = keys2[vcz2_only_idx]
    all_keys[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = keys1[exact_i1]
    if n_py > 0:
        all_keys[n_v1 + n_v2 + n_ex :] = [r[0] for r in python_output]

    all_tb = np.zeros(n_out, dtype=np.int8)
    all_tb[n_v1 : n_v1 + n_v2] = 1  # vcz2-only: tiebreak=1
    if n_py > 0:
        all_tb[n_v1 + n_v2 + n_ex :] = [r[1] for r in python_output]

    sort_order = np.lexsort([all_tb, all_keys])

    # Contig and position arrays
    all_contig = np.empty(n_out, dtype=vc1.dtype)
    all_contig[:n_v1] = vc1[vcz1_only_idx]
    all_contig[n_v1 : n_v1 + n_v2] = vc2[vcz2_only_idx]
    all_contig[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = vc1[exact_i1]
    if n_py > 0:
        all_contig[n_v1 + n_v2 + n_ex :] = [r[2] for r in python_output]

    all_pos = np.empty(n_out, dtype=vp1.dtype)
    all_pos[:n_v1] = vp1[vcz1_only_idx]
    all_pos[n_v1 : n_v1 + n_v2] = vp2[vcz2_only_idx]
    all_pos[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = vp1[exact_i1]
    if n_py > 0:
        all_pos[n_v1 + n_v2 + n_ex :] = [r[3] for r in python_output]

    # Length array (only when both stores have variant_length)
    if has_length:
        all_length: np.ndarray | None = np.empty(n_out, dtype=vl1.dtype)
        all_length[:n_v1] = vl1[vcz1_only_idx]
        all_length[n_v1 : n_v1 + n_v2] = vl2[vcz2_only_idx]
        all_length[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = np.maximum(
            vl1[exact_i1], vl2[exact_i2]
        )
        if n_py > 0:
            all_length[n_v1 + n_v2 + n_ex :] = [r[5] for r in python_output]
    else:
        all_length = None

    # ID array (only when both stores have variant_id)
    if has_id:
        _str_dt = np.dtypes.StringDType()
        all_id: np.ndarray | None = np.empty(n_out, dtype=_str_dt)
        all_id[:n_v1] = vi1[vcz1_only_idx].astype(_str_dt)
        all_id[n_v1 : n_v1 + n_v2] = vi2[vcz2_only_idx].astype(_str_dt)
        all_id[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = [
            _merge_ids(str(vi1[i1]), str(vi2[i2])) for i1, i2 in zip(exact_i1, exact_i2)
        ]
        if n_py > 0:
            all_id[n_v1 + n_v2 + n_ex :] = [r[6] for r in python_output]
    else:
        all_id = None

    # Quality array (only when both stores have variant_quality)
    if has_quality:
        all_quality: np.ndarray | None = np.empty(n_out, dtype=vq1.dtype)
        all_quality[:n_v1] = vq1[vcz1_only_idx]
        all_quality[n_v1 : n_v1 + n_v2] = vq2[vcz2_only_idx]
        all_quality[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = np.fmax(
            vq1[exact_i1], vq2[exact_i2]
        )
        if n_py > 0:
            all_quality[n_v1 + n_v2 + n_ex :] = [r[7] for r in python_output]
    else:
        all_quality = None

    # Filter array (only when both stores have variant_filter with identical filter_id)
    if has_filter:
        n_filters = vf1.shape[1]
        all_filter: np.ndarray | None = np.zeros((n_out, n_filters), dtype=bool)
        all_filter[:n_v1] = vf1[vcz1_only_idx]
        all_filter[n_v1 : n_v1 + n_v2] = vf2[vcz2_only_idx]
        all_filter[n_v1 + n_v2 : n_v1 + n_v2 + n_ex] = vf1[exact_i1] | vf2[exact_i2]
        if n_py > 0:
            for k, r in enumerate(python_output):
                all_filter[n_v1 + n_v2 + n_ex + k] = r[8]
    else:
        all_filter = None

    # Allele array: width = max alleles across all sources
    max_alleles = max(
        va1.shape[1] if n1 > 0 else 1,
        va2.shape[1] if n2 > 0 else 1,
        max((len(r[4]) for r in python_output), default=1),
    )

    all_allele = np.full((n_out, max_alleles), STR_FILL, dtype=STRING_DTYPE_NAME)
    all_allele[:n_v1, : va1.shape[1]] = va1[vcz1_only_idx]
    all_allele[n_v1 : n_v1 + n_v2, : va2.shape[1]] = va2[vcz2_only_idx]
    all_allele[n_v1 + n_v2 : n_v1 + n_v2 + n_ex, : va1.shape[1]] = va1[exact_i1]
    for k, rec in enumerate(python_output):
        for j, a in enumerate(rec[4]):
            all_allele[n_v1 + n_v2 + n_ex + k, j] = a

    out_length = all_length[sort_order] if all_length is not None else None
    out_id = all_id[sort_order] if all_id is not None else None
    out_quality = all_quality[sort_order] if all_quality is not None else None
    out_filter = all_filter[sort_order] if all_filter is not None else None
    return (
        all_contig[sort_order],
        all_pos[sort_order],
        out_length,
        out_id,
        out_quality,
        out_filter,
        all_allele[sort_order],
    )


def create(vcz1, vcz2, vcz_out, *, show_progress=False, backend_storage=None) -> None:
    """Create a new, empty store vcz_out using merged variants from vcz1 and vcz2
    using -m none semantics with stable variant ordering.

    Both stores must have identical contig_id arrays. Output contains all variants
    from both stores; variants at the same position whose alt sets overlap are merged
    into a single record with combined alleles. Only fixed variant fields are
    written; other variant fields are not included, and sample and call
    fields are included, but with no samples.

    When two records are merged: variant_id are joined with ";", variant_quality takes
    the max (ignoring missing), variant_filter is the union. variant_filter requires
    identical filter_id arrays. Optional fields (variant_length, variant_id,
    variant_quality, variant_filter) are included only when both input stores contain
    them.
    """
    root1 = open_zarr(vcz1, mode="r", backend_storage=backend_storage)
    root2 = zarr.open(vcz2, mode="r")

    if not np.all(root1["contig_id"][:] == root2["contig_id"][:]):
        raise ValueError("contig_id fields must be identical")

    (
        out_contig,
        out_position,
        out_length,
        out_id,
        out_quality,
        out_filter,
        out_allele,
    ) = _compute_merged_variants(root1, root2, show_progress=show_progress)

    n_variants = out_contig.shape[0]
    variants_chunk_size = root1["variant_contig"].chunks[0]

    out_root = zarr.open(vcz_out, mode="w", zarr_format=root1.metadata.zarr_format)
    out_root.attrs.update(root1.attrs)

    # copy direct from vcz1
    vcz1_copy_vars = [
        var
        for var in root1.keys()
        if var.startswith("contig_") or var.startswith("filter_")
    ]
    copy_store(vcz1, vcz_out, array_keys=vcz1_copy_vars)

    arr = root1["variant_contig"]
    create_group_array(
        out_root,
        "variant_contig",
        data=out_contig,
        shape=out_contig.shape,
        dtype=arr.dtype,
        chunks=(variants_chunk_size,),
        compressor=get_compressor_config(arr),
        dimension_names=["variants"],
    )
    arr = root1["variant_position"]
    create_group_array(
        out_root,
        "variant_position",
        data=out_position,
        shape=out_position.shape,
        dtype=arr.dtype,
        chunks=(variants_chunk_size,),
        compressor=get_compressor_config(arr),
        dimension_names=["variants"],
    )
    if out_length is not None:
        arr = root1["variant_length"]
        create_group_array(
            out_root,
            "variant_length",
            data=out_length,
            shape=out_length.shape,
            dtype=arr.dtype,
            chunks=(variants_chunk_size,),
            compressor=get_compressor_config(arr),
            dimension_names=["variants"],
        )

    if out_id is not None:
        arr = root1["variant_id"]
        create_group_array(
            out_root,
            "variant_id",
            data=out_id,
            shape=out_id.shape,
            dtype=STRING_DTYPE_NAME,
            chunks=(variants_chunk_size,),
            compressor=get_compressor_config(arr),
            dimension_names=["variants"],
        )
        arr = root1["variant_id_mask"]
        create_group_array(
            out_root,
            "variant_id_mask",
            data=out_id == ".",
            shape=out_id.shape,
            dtype=arr.dtype,
            chunks=(variants_chunk_size,),
            compressor=get_compressor_config(arr),
            dimension_names=["variants"],
        )
    if out_quality is not None:
        arr = root1["variant_quality"]
        create_group_array(
            out_root,
            "variant_quality",
            data=out_quality,
            shape=out_quality.shape,
            dtype=arr.dtype,
            chunks=(variants_chunk_size,),
            compressor=get_compressor_config(arr),
            dimension_names=["variants"],
        )
    if out_filter is not None:
        arr = root1["variant_filter"]
        create_group_array(
            out_root,
            "variant_filter",
            data=out_filter,
            shape=out_filter.shape,
            dtype=arr.dtype,
            chunks=(variants_chunk_size,) + arr.chunks[1:],
            compressor=get_compressor_config(arr),
            dimension_names=["variants", "filters"],
        )

    arr = root1["variant_allele"]
    create_group_array(
        out_root,
        "variant_allele",
        data=out_allele,
        shape=out_allele.shape,
        dtype=STRING_DTYPE_NAME,
        chunks=(variants_chunk_size,) + arr.chunks[1:],
        compressor=get_compressor_config(arr),
        dimension_names=["variants", "alleles"],
    )

    # create empty sample and call arrays
    for var in root1.keys():
        if var.startswith("call_"):
            arr = root1[var]
            shape = (n_variants, 0) + arr.shape[2:]
            chunks = (variants_chunk_size,) + arr.chunks[1:]
            create_empty_group_array(
                out_root,
                var,
                shape=shape,
                dtype=arr.dtype,
                chunks=chunks,
                compressor=get_compressor_config(arr),
                dimension_names=array_dims(arr),
            )
        elif var == "sample_id":
            arr = root1[var]
            shape = (0,)
            chunks = arr.chunks
            create_empty_group_array(
                out_root,
                var,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=arr.chunks,
                compressor=get_compressor_config(arr),
                dimension_names=array_dims(arr),
            )

    if out_length is not None:
        indexer = VcfZarrIndexer(vcz_out)
        indexer.create_index()
