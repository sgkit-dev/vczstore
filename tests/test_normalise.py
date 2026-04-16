import pathlib
import re

import numpy as np
import pytest
import zarr

from vczstore.zarr_impl import index_variants, normalise, variant_alleles_are_equivalent

from .utils import compare_vcf_and_vcz, convert_vcf_to_vcz


def create_variant_only_vcz(variant_contig, variant_position, variant_allele):
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store)
    root.create_array(
        name="contig_id",
        data=np.unique(variant_contig).astype(str),
        dimension_names=["contigs"],
    )
    root.create_array(
        name="variant_contig",
        data=np.array(variant_contig, dtype=np.int32),
        dimension_names=["variants"],
    )
    root.create_array(
        name="variant_position",
        data=np.array(variant_position, dtype=np.int32),
        dimension_names=["variants"],
    )
    root.create_array(
        name="variant_allele",
        data=np.array(variant_allele, dtype="T"),
        dimension_names=["variants", "alleles"],
    )
    root.create_array(
        name="sample_id", data=np.array([], dtype="T"), dimension_names=["samples"]
    )
    return store


@pytest.mark.parametrize(
    ("variant_allele1", "variant_allele2", "expected"),
    [
        ([], [], True),
        (["A"], ["A"], True),
        (["A", "T"], ["A", "T"], True),
        (["A", ""], ["A"], True),
        (["A", "."], ["A"], True),
        (["A", "T", "."], ["A", "T", ".", "."], True),
        (["A", "T", "."], ["A", "T", "", ""], True),
        (["."], ["A"], False),
        (["A", ".", "T"], ["A", "T"], False),
        (["A", ".", "T"], ["A", "T", "."], False),
    ],
)
def test_variant_alleles_are_equivalent(variant_allele1, variant_allele2, expected):
    assert variant_alleles_are_equivalent(variant_allele1, variant_allele2) == expected


def test_index_variants__success_subset():
    vcz1 = create_variant_only_vcz(
        [0, 0, 0], [1, 2, 3], [["A", "T"], ["C", "G"], ["T", "A"]]
    )
    vcz2 = create_variant_only_vcz([0, 0], [1, 3], [["A", "T"], ["T", "A"]])
    np.testing.assert_array_equal(index_variants(vcz1, vcz2), [True, False, True])


def test_index_variants__success_repeated_site():
    # example where multi-allelic sites are split to biallelic
    vcz1 = create_variant_only_vcz(
        [0, 0, 0], [1, 1, 3], [["A", "T"], ["A", "C"], ["T", "A"]]
    )
    vcz2 = create_variant_only_vcz([0, 0], [1, 1], [["A", "T"], ["A", "C"]])
    np.testing.assert_array_equal(index_variants(vcz1, vcz2), [True, True, False])


def test_index_variants__order_mismatch():
    # example where multi-allelic sites are split to biallelic
    vcz1 = create_variant_only_vcz(
        [0, 0, 0], [1, 1, 3], [["A", "T"], ["A", "C"], ["T", "A"]]
    )
    # note that the allele ordering is different for contig 0, position 1
    vcz2 = create_variant_only_vcz([0, 0], [1, 1], [["A", "C"], ["A", "T"]])
    with pytest.raises(
        match=re.escape(
            "Variant not in (or out of order in) first vcz: "
            "variant_contig=0, variant_position=1, variant_allele=['A', 'T']"
        )
    ):
        index_variants(vcz1, vcz2)


def test_index_variants__new_variant():
    vcz1 = create_variant_only_vcz(
        [0, 0, 0], [2, 3, 4], [["A", "T"], ["C", "G"], ["T", "A"]]
    )
    vcz2 = create_variant_only_vcz([0, 0, 0], [1, 2], [["A", "."], ["A", "T"]])
    with pytest.raises(
        match=re.escape(
            "Variant not in (or out of order in) first vcz: "
            "variant_contig=0, variant_position=1, variant_allele=['A', '.']"
        )
    ):
        index_variants(vcz1, vcz2)


def test_index_variants__new_variant_at_end():
    vcz1 = create_variant_only_vcz(
        [0, 0, 0], [1, 2, 3], [["A", "T"], ["C", "G"], ["T", "A"]]
    )
    vcz2 = create_variant_only_vcz([0, 0, 0], [1, 4], [["A", "T"], ["G", "A"]])
    with pytest.raises(
        match=re.escape(
            "Variant not in first vcz: "
            "variant_contig=0, variant_position=4, variant_allele=['G', 'A']"
        )
    ):
        index_variants(vcz1, vcz2)


def test_index_variants__new_allele():
    vcz1 = create_variant_only_vcz(
        [0, 0, 0], [1, 2, 3], [["A", "T"], ["C", "G"], ["T", "A"]]
    )
    # variant at contig 0, position 3 has different alleles
    vcz2 = create_variant_only_vcz([0, 0], [1, 3], [["A", "T"], ["T", "G"]])
    with pytest.raises(
        match=re.escape(
            "Variant not in first vcz: "
            "variant_contig=0, variant_position=3, variant_allele=['T', 'G']"
        )
    ):
        index_variants(vcz1, vcz2)


# bcftools merge tests/data/vcf/alleles-variants.vcf.gz \
#  tests/data/vcf/alleles-1.vcf.gz \
#  -o tests/data/vcf/alleles-1-norm.vcf.gz -W=csi
def test_normalise(tmp_path):
    vcz0 = convert_vcf_to_vcz("alleles-variants.vcf.gz", tmp_path, ploidy=2)
    vcz1 = convert_vcf_to_vcz("alleles-1.vcf.gz", tmp_path)

    vcz1_norm = pathlib.Path(tmp_path) / "alleles-1-norm.vcz"

    normalise(vcz0, vcz1, vcz1_norm)

    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version",
        "alleles-1-norm.vcf.gz",
        "view --no-version",
        vcz1_norm,
    )
