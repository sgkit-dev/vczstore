import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from vczstore.create import _can_merge_variants, _merge_alts, _merge_ids, create

from .utils import make_vcz


@pytest.mark.parametrize(
    ("allele_a", "allele_b", "expected"),
    [
        # same ref + same alt
        (["A", "T"], ["A", "T"], True),
        # shared alt
        (["A", "T", "G"], ["A", "C", "G"], True),
        # disjoint alts
        (["A", "T"], ["A", "C"], False),
        # different ref
        (["A", "T"], ["C", "T"], False),
        # ref-only always merges
        (["A", "."], ["A", "T"], True),
        (["A", "T"], ["A", "."], True),
        # padding stripped before comparison
        (["A", "T", ""], ["A", "T"], True),
        (["A", "T", "."], ["A", "T"], True),
    ],
)
def test_can_create(allele_a, allele_b, expected):
    assert _can_merge_variants(np.array(allele_a), np.array(allele_b)) == expected


@pytest.mark.parametrize(
    ("alts1", "alts2", "expected"),
    [
        (["T"], ["T"], ["T"]),
        (["T"], ["C"], ["T", "C"]),
        (["T", "G"], ["C", "G"], ["T", "G", "C"]),
        (["T"], ["T", "C"], ["T", "C"]),
        (["T", "C"], ["T"], ["T", "C"]),
        (["."], ["T"], ["T"]),
        (["T"], ["."], ["T"]),
        (["."], ["."], ["."]),
    ],
)
def test_merge_alts(alts1, alts2, expected):
    assert _merge_alts(alts1, alts2) == expected


def test_create__no_match():
    # Disjoint alts at same position → 2 output variants
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    vcz2 = make_vcz([0], [100], [["A", "C"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_position"][:], [100, 100])
    assert root["variant_allele"][0, 0] == "A"
    assert root["variant_allele"][0, 1] == "T"
    assert root["variant_allele"][1, 0] == "A"
    assert root["variant_allele"][1, 1] == "C"


def test_create__identical():
    # Same record from both stores → 1 output variant
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    vcz2 = make_vcz([0], [100], [["A", "T"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_position"][:], [100])
    assert_array_equal(root["variant_allele"][:, :2], [["A", "T"]])


def test_create__overlapping_alts():
    # Shared alt G → 1 merged variant with combined alleles [A, T, G, C]
    vcz1 = make_vcz([0], [100], [["A", "T", "G"]])
    vcz2 = make_vcz([0], [100], [["A", "C", "G"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_position"][:], [100])
    alleles = [a for a in root["variant_allele"][0] if a != ""]
    assert alleles == ["A", "T", "G", "C"]


def test_create__stable_ordering():
    # Mirrors test_merge_pairwise: [A/T, A/C] + [A/G, A/C] → [A/T, A/G, A/C]
    vcz1 = make_vcz([0, 0], [100, 100], [["A", "T"], ["A", "C"]])
    vcz2 = make_vcz([0, 0], [100, 100], [["A", "G"], ["A", "C"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_position"][:], [100, 100, 100])
    alts = [root["variant_allele"][i, 1] for i in range(3)]
    assert alts == ["T", "G", "C"]


def test_create__different_positions():
    # Variants at different positions pass through unchanged
    vcz1 = make_vcz([0, 0], [100, 200], [["A", "T"], ["C", "G"]])
    vcz2 = make_vcz([0, 0], [150, 200], [["G", "T"], ["C", "G"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_position"][:], [100, 150, 200])


def test_create__multi_contig():
    # Variants on multiple contigs
    vcz1 = make_vcz([0, 1], [100, 200], [["A", "T"], ["C", "G"]])
    vcz2 = make_vcz([0, 1], [100, 300], [["A", "T"], ["G", "A"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_contig"][:], [0, 1, 1])
    assert_array_equal(root["variant_position"][:], [100, 200, 300])


def test_create__ordering_conflict_raises():
    # vcz1=[A/T, A/C] and vcz2=[A/C, A/T] at the same position.
    # The greedy step pairs A/T↔A/T and A/C↔A/C, creating ordering constraints:
    # vcz1 requires A/T before A/C, vcz2 requires A/C before A/T — a cycle → ValueError.
    vcz1 = make_vcz([0, 0], [100, 100], [["A", "T"], ["A", "C"]])
    vcz2 = make_vcz([0, 0], [100, 100], [["A", "C"], ["A", "T"]])
    vcz_out = zarr.storage.MemoryStore()
    with pytest.raises(ValueError, match="ordering conflict"):
        create(vcz1, vcz2, vcz_out)


def test_create__variant_length_max():
    # Merged variant takes max of the two stores' lengths
    vcz1 = make_vcz([0], [100], [["A", "T"]], variant_length=[5])
    vcz2 = make_vcz([0], [100], [["A", "T"]], variant_length=[3])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_length"][:], [5])


def test_create__variant_length_passthrough():
    # Variants from only one store pass their length through unchanged
    vcz1 = make_vcz([0], [100], [["A", "T"]], variant_length=[4])
    vcz2 = make_vcz([0], [200], [["C", "G"]], variant_length=[7])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_length"][:], [4, 7])


def test_create__variant_length_absent():
    # Neither store has variant_length → output has none either
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    vcz2 = make_vcz([0], [100], [["A", "T"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert "variant_length" not in root


@pytest.mark.parametrize(
    ("id1", "id2", "expected"),
    [
        (".", ".", "."),
        ("rs1", ".", "rs1"),
        (".", "rs2", "rs2"),
        ("rs1", "rs2", "rs1;rs2"),
        ("rs1", "rs1", "rs1"),
    ],
)
def test_merge_ids(id1, id2, expected):
    assert _merge_ids(id1, id2) == expected


def test_create__variant_id_joined():
    # Merged variant joins IDs with ";"
    vcz1 = make_vcz([0], [100], [["A", "T"]], variant_id=["rs1"])
    vcz2 = make_vcz([0], [100], [["A", "T"]], variant_id=["rs2"])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert root["variant_id"][0] == "rs1;rs2"
    assert root["variant_id_mask"][0] == False  # noqa: E712


def test_create__variant_id_missing_dropped():
    # "." ID from one side is dropped when joining
    vcz1 = make_vcz([0], [100], [["A", "T"]], variant_id=["rs1"])
    vcz2 = make_vcz([0], [100], [["A", "T"]], variant_id=["."])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert root["variant_id"][0] == "rs1"
    assert root["variant_id_mask"][0] == False  # noqa: E712


def test_create__variant_id_passthrough():
    # Variants from only one store keep their ID unchanged
    vcz1 = make_vcz([0], [100], [["A", "T"]], variant_id=["rs1"])
    vcz2 = make_vcz([0], [200], [["C", "G"]], variant_id=["rs2"])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_id"][:], ["rs1", "rs2"])
    assert_array_equal(root["variant_id_mask"][:], [False, False])


def test_create__variant_id_absent():
    # Neither store has variant_id → output has none either
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    vcz2 = make_vcz([0], [100], [["A", "T"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert "variant_id" not in root
    assert "variant_id_mask" not in root


def test_create__variant_quality_max():
    # Merged variant takes max QUAL; missing (NaN) treated as absent
    vcz1 = make_vcz(
        [0, 0],
        [100, 200],
        [["A", "T"], ["G", "C"]],
        variant_quality=[10.0, float("nan")],
    )
    vcz2 = make_vcz(
        [0, 0], [100, 200], [["A", "T"], ["G", "C"]], variant_quality=[5.0, 8.0]
    )
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert root["variant_quality"][0] == pytest.approx(10.0)
    assert root["variant_quality"][1] == pytest.approx(8.0)


def test_create__variant_quality_passthrough():
    # Variants from only one store keep their QUAL unchanged
    vcz1 = make_vcz([0], [100], [["A", "T"]], variant_quality=[30.0])
    vcz2 = make_vcz([0], [200], [["C", "G"]], variant_quality=[20.0])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    np.testing.assert_allclose(root["variant_quality"][:], [30.0, 20.0])


def test_create__variant_quality_absent():
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    vcz2 = make_vcz([0], [100], [["A", "T"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert "variant_quality" not in root


def test_create__variant_filter_union():
    # Merged variant filter is the union (OR) of both stores' filters
    # filter layout: [PASS, q10] → columns 0, 1
    filt_id = ["PASS", "q10"]
    vcz1 = make_vcz(
        [0],
        [100],
        [["A", "T"]],
        variant_filter=[[True, False]],
        filter_id=filt_id,
    )
    vcz2 = make_vcz(
        [0],
        [100],
        [["A", "T"]],
        variant_filter=[[False, True]],
        filter_id=filt_id,
    )
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_filter"][0], [True, True])
    assert_array_equal(root["filter_id"][:], filt_id)


def test_create__variant_filter_passthrough():
    # Variants from only one store keep their filter unchanged
    filt_id = ["PASS", "q10"]
    vcz1 = make_vcz(
        [0],
        [100],
        [["A", "T"]],
        variant_filter=[[True, False]],
        filter_id=filt_id,
    )
    vcz2 = make_vcz(
        [0],
        [200],
        [["C", "G"]],
        variant_filter=[[False, True]],
        filter_id=filt_id,
    )
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert_array_equal(root["variant_filter"][:], [[True, False], [False, True]])


def test_create__variant_filter_absent():
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    vcz2 = make_vcz([0], [100], [["A", "T"]])
    vcz_out = zarr.storage.MemoryStore()
    create(vcz1, vcz2, vcz_out)
    root = zarr.open(vcz_out)
    assert "variant_filter" not in root
    assert "filter_id" not in root


def test_create__different_filter_ids_raises():
    vcz1 = make_vcz(
        [0],
        [100],
        [["A", "T"]],
        variant_filter=[[True]],
        filter_id=["PASS"],
    )
    vcz2 = make_vcz(
        [0],
        [100],
        [["A", "T"]],
        variant_filter=[[True]],
        filter_id=["q10"],
    )
    vcz_out = zarr.storage.MemoryStore()
    with pytest.raises(ValueError, match="filter_id"):
        create(vcz1, vcz2, vcz_out)


def test_create__different_contig_ids_raises():
    vcz1 = make_vcz([0], [100], [["A", "T"]])
    # manually create a vcz with different contig_id
    vcz2 = zarr.storage.MemoryStore()
    root = zarr.create_group(store=vcz2)
    root.create_array(name="contig_id", data=np.array(["X"]))
    root.create_array(name="variant_contig", data=np.array([0], dtype=np.int32))
    root.create_array(name="variant_position", data=np.array([100], dtype=np.int32))
    root.create_array(name="variant_allele", data=np.array([["A", "T"]]))
    vcz_out = zarr.storage.MemoryStore()
    with pytest.raises(ValueError, match="contig_id"):
        create(vcz1, vcz2, vcz_out)
