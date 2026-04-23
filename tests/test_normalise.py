import re

import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from vczstore.append import append
from vczstore.normalise import (
    index_variants,
    normalise,
    remap_genotypes,
    variant_alleles_are_equivalent,
)

from .utils import compare_vcf_and_vcz, convert_vcf_to_vcz, convert_vcf_to_vcz_icechunk


# TODO: replace this with make_vcz in vcztools/bio2zarr
# see https://github.com/sgkit-dev/vczstore/issues/37
def make_vcz(
    variant_contig,
    variant_position,
    alleles,
    *,
    sample_id=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    call_genotype=None,
    call_fields=None,
    call_field_dims=None,
    ploidy=2,
):

    variant_contig = np.asarray(variant_contig, dtype=np.int32)
    variant_position = np.asarray(variant_position, dtype=np.int32)
    num_variants = variant_contig.shape[0]
    max_alleles = max(len(a) for a in alleles) if len(alleles) > 0 else 1
    allele_array = np.full((num_variants, max_alleles), "", dtype="<U16")
    for i, row in enumerate(alleles):
        for j, a in enumerate(row):
            allele_array[i, j] = a

    num_samples = len(sample_id) if sample_id is not None else 0

    v_chunk = variants_chunk_size if variants_chunk_size is not None else num_variants
    v_chunk = max(v_chunk, 1)
    s_chunk = (
        samples_chunk_size if samples_chunk_size is not None else max(num_samples, 1)
    )

    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store)
    root.create_array(
        name="contig_id",
        data=np.unique(variant_contig).astype(str),
        dimension_names=["contigs"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        name="variant_contig",
        data=variant_contig,
        chunks=(v_chunk,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        name="variant_position",
        data=variant_position,
        chunks=(v_chunk,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        name="variant_allele",
        data=allele_array,
        chunks=(v_chunk, max_alleles),
        dimension_names=["variants", "alleles"],
        compressors=None,
        filters=None,
    )
    if sample_id is None:
        sample_id_arr = np.array([], dtype="<U64")
    else:
        sample_id_arr = np.asarray(sample_id, dtype="<U64")
    root.create_array(
        name="sample_id",
        data=sample_id_arr,
        dimension_names=["samples"],
        compressors=None,
        filters=None,
    )
    if call_genotype is not None:
        call_genotype = np.asarray(call_genotype, dtype=np.int8)
        root.create_array(
            name="call_genotype",
            data=call_genotype,
            chunks=(v_chunk, s_chunk, ploidy),
            dimension_names=["variants", "samples", "ploidy"],
            compressors=None,
            filters=None,
        )
    if call_fields is not None:
        for name, data in call_fields.items():
            if data.ndim == 2:
                chunks = (v_chunk, s_chunk)
            elif data.ndim == 3:
                chunks = (v_chunk, s_chunk, data.shape[2])
            else:
                raise ValueError("call field arrays must be 2D or 3D")
            root.create_array(
                name=f"call_{name}",
                data=data,
                chunks=chunks,
                dimension_names=call_field_dims[name],
                compressors=None,
                filters=None,
            )
    return store


@pytest.mark.parametrize(
    (
        "variant_allele1",
        "variant_allele2",
        "expected_matched",
        "expected_mapping",
        "expected_updated",
    ),
    [
        (["A"], ["A"], True, None, None),
        (["A", "T"], ["A", "T"], True, None, None),
        (["A", ""], ["A"], True, None, None),
        (["A", "."], ["A"], True, None, None),
        (["A", "T", "."], ["A", "T", ".", "."], True, None, None),
        (["A", "T", "."], ["A", "T", "", ""], True, None, None),
        # overlapping alt alleles
        (["A", "C", "T"], ["A", "T"], True, [0, 2], None),
        # new alt alleles
        (["A", "T"], ["A", "T", "C"], True, [0, 1, 2], ["A", "T", "C"]),
        (["A", "T"], ["A", "C", "T"], True, [0, 2, 1], ["A", "T", "C"]),
        # indels
        (["AT", "A"], ["A", "AT"], False, None, None),
        # different alt allele
        (["A", "C"], ["A", "T"], False, None, None),
    ],
)
def test_variant_alleles_are_equivalent(
    variant_allele1,
    variant_allele2,
    expected_matched,
    expected_mapping,
    expected_updated,
):
    matched, mapping, updated = variant_alleles_are_equivalent(
        np.array(variant_allele1), np.array(variant_allele2)
    )
    assert matched == expected_matched
    assert_array_equal(mapping, expected_mapping)
    assert_array_equal(updated, expected_updated)


def test_remap_genotypes__no_op():
    gt = np.array([[[0, 1]], [[1, 0]]], dtype=np.int8)
    original = gt.copy()
    remap_genotypes(gt, [], [])
    assert_array_equal(gt, original)


def test_remap_genotypes__single_variant():
    # alleles [A, T, C] remapped to [A, C, T]: mapping is [0, 2, 1]
    gt = np.array([[[0, 1]], [[2, 1]]], dtype=np.int8)
    remap_genotypes(gt, [1], [np.array([0, 2, 1])])
    assert_array_equal(gt, [[[0, 1]], [[1, 2]]])


def test_remap_genotypes__multiple_variants():
    gt = np.array([[[1, 2]], [[3, 0]]], dtype=np.int8)
    # variant 0: vcz2=[A, C, T], vcz1=[A, T, C]: mapping [0, 2, 1]
    # variant 1: vcz2=[A, G, T, C], vcz1=[A, T, C, G]: mapping [0, 3, 1, 2]
    remap_genotypes(gt, [0, 1], [np.array([0, 2, 1]), np.array([0, 3, 1, 2])])
    assert_array_equal(gt, [[[2, 1]], [[2, 0]]])


def test_index_variants__success_subset():
    vcz1 = make_vcz([0, 0, 0], [1, 2, 3], [["A", "T"], ["C", "G"], ["T", "A"]])
    vcz2 = make_vcz([0, 0], [1, 3], [["A", "T"], ["T", "A"]])
    index, *_ = index_variants(vcz1, vcz2)
    assert_array_equal(index, [True, False, True])


def test_index_variants__success_repeated_site():
    # example where multi-allelic sites are split to biallelic
    vcz1 = make_vcz([0, 0, 0], [1, 1, 3], [["A", "T"], ["A", "C"], ["T", "A"]])
    vcz2 = make_vcz([0, 0], [1, 1], [["A", "T"], ["A", "C"]])
    index, *_ = index_variants(vcz1, vcz2)
    assert_array_equal(index, [True, True, False])


def test_index_variants__order_mismatch():
    # example where multi-allelic sites are split to biallelic
    vcz1 = make_vcz([0, 0, 0], [1, 1, 3], [["A", "T"], ["A", "C"], ["T", "A"]])
    # note that the allele ordering is different for contig 0, position 1
    vcz2 = make_vcz([0, 0], [1, 1], [["A", "C"], ["A", "T"]])
    with pytest.raises(
        match=re.escape(
            "Variant in vcz2 not found in vcz1 (or vcz2 is out of order): "
            "variant_contig=0, variant_position=1, variant_allele=['A', 'T']"
        )
    ):
        index_variants(vcz1, vcz2)


def test_index_variants__new_variant():
    vcz1 = make_vcz([0, 0, 0], [2, 3, 4], [["A", "T"], ["C", "G"], ["T", "A"]])
    vcz2 = make_vcz([0, 0, 0], [1, 2], [["A", "."], ["A", "T"]])
    with pytest.raises(
        match=re.escape(
            "Variant in vcz2 not found in vcz1 (or vcz2 is out of order): "
            "variant_contig=0, variant_position=1, variant_allele=['A', '.']"
        )
    ):
        index_variants(vcz1, vcz2)


def test_index_variants__new_variant_at_end():
    vcz1 = make_vcz([0, 0, 0], [1, 2, 3], [["A", "T"], ["C", "G"], ["T", "A"]])
    vcz2 = make_vcz([0, 0, 0], [1, 4], [["A", "T"], ["G", "A"]])
    with pytest.raises(
        match=re.escape(
            "Variant not in first vcz: "
            "variant_contig=0, variant_position=4, variant_allele=['G', 'A']"
        )
    ):
        index_variants(vcz1, vcz2)


def test_index_variants__new_allele():
    vcz1 = make_vcz([0, 0, 0], [1, 2, 3], [["A", "T"], ["C", "G"], ["T", "A"]])
    # variant at contig 0, position 3 has different alleles
    vcz2 = make_vcz([0, 0], [1, 3], [["A", "T"], ["T", "G"]])
    with pytest.raises(
        match=re.escape(
            "Variant not in first vcz: "
            "variant_contig=0, variant_position=3, variant_allele=['T', 'G']"
        )
    ):
        index_variants(vcz1, vcz2)


@pytest.mark.parametrize("variants_chunk_size", [None, 1, 3, 4, 5, 10])
@pytest.mark.parametrize("io_concurrency", [1, 4])
def test_normalise(variants_chunk_size, io_concurrency):
    vcz1 = make_vcz(
        variant_contig=[0, 0, 0, 0, 0, 0, 0, 0, 0],
        variant_position=[1, 2, 3, 4, 4, 5, 5, 6, 7],
        alleles=[
            ["A", "T"],
            ["A", "C"],
            ["A", "G"],
            ["A", "C"],
            ["A", "G", "T"],
            ["A", "G"],
            ["A", "T", "C"],
            ["A", "G", "T"],
            ["A", "T", "G"],
        ],
        variants_chunk_size=variants_chunk_size,
    )

    vcz2 = make_vcz(
        variant_contig=[0, 0, 0, 0, 0, 0],
        variant_position=[1, 2, 4, 5, 6, 7],
        alleles=[
            ["A", "T"],
            ["A", "C"],
            ["A", "C"],
            ["A", "T", "C"],
            ["A", "G"],
            ["A", "G", "T"],  # order different to vcz1
        ],
        sample_id=["S1"],
        call_genotype=[[[0, 0]], [[0, 1]], [[0, 0]], [[1, 1]], [[0, 1]], [[0, 2]]],
    )

    vcz2_norm = zarr.storage.MemoryStore()

    normalise(vcz1, vcz2, vcz2_norm, io_concurrency=io_concurrency)

    root1 = zarr.open(vcz1)
    root_norm = zarr.open(vcz2_norm)

    assert_array_equal(root_norm["variant_contig"][:], root1["variant_contig"][:])
    assert_array_equal(root_norm["variant_position"][:], root1["variant_position"][:])
    assert_array_equal(root_norm["variant_allele"][:], root1["variant_allele"][:])
    assert_array_equal(root_norm["sample_id"][:], ["S1"])
    assert_array_equal(
        root_norm["call_genotype"][:],
        [
            [[0, 0]],
            [[0, 1]],
            [[-1, -1]],
            [[0, 0]],
            [[-1, -1]],
            [[-1, -1]],
            [[1, 1]],
            [[0, 1]],
            [[0, 1]],  # remapped to vcz1 order
        ],
    )


def test_normalise__other_call_fields_not_implemented():
    vcz1 = make_vcz(
        variant_contig=[0],
        variant_position=[1],
        alleles=[["A", "T", "G"]],
    )

    vcz2 = make_vcz(
        variant_contig=[0],
        variant_position=[1],
        alleles=[["A", "G", "T"]],  # order different to vcz1
        sample_id=["S1"],
        call_genotype=[[[0, 2]]],
        call_fields={"AD": np.array([[[10, -1, 20]]], dtype=np.int8)},
        call_field_dims={"AD": ["variants", "samples", "alleles"]},
    )

    vcz2_norm = zarr.storage.MemoryStore()

    with pytest.raises(NotImplementedError):
        normalise(vcz1, vcz2, vcz2_norm)


def test_normalise_and_append(tmp_path):
    vcz0 = convert_vcf_to_vcz("sample-variants.vcf.gz", tmp_path, ploidy=2)
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path)
    vcz1_norm = zarr.storage.MemoryStore()

    normalise(vcz0, vcz1, vcz1_norm)

    append(vcz0, vcz1_norm)

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", "sample-part1.vcf.gz", "view --no-version", vcz0
    )


def test_normalise_and_append_icechunk(tmp_path):
    pytest.importorskip("icechunk")
    from vczstore.icechunk_utils import icechunk_transaction

    # note that vcz0 is in icechunk, but the others needn't be
    vcz0 = convert_vcf_to_vcz_icechunk("sample-variants.vcf.gz", tmp_path, ploidy=2)
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path)
    vcz1_norm = zarr.storage.MemoryStore()

    with icechunk_transaction(vcz0, "main", message="append") as store:
        normalise(store, vcz1, vcz1_norm)
        append(store, vcz1_norm)

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version",
        "sample-part1.vcf.gz",
        "view --no-version --zarr-backend-storage icechunk",
        vcz0,
    )
