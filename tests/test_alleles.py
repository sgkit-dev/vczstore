# bcftools view tests/data/vcf/alleles-1.vcf --write-index=csi \
#  -o tests/data/vcf/alleles-1.vcf.gz
# bcftools view tests/data/vcf/alleles-2.vcf --write-index=csi \
#  -o tests/data/vcf/alleles-2.vcf.gz

# merging
# bcftools merge tests/data/vcf/alleles-1.vcf.gz tests/data/vcf/alleles-2.vcf.gz
# bcftools merge tests/data/vcf/alleles-1.vcf.gz tests/data/vcf/alleles-2.vcf.gz -O z \
#  > tests/data/vcf/alleles-merged.vcf.gz

import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from vczstore.allele_harmonisation import harmonise_alleles
from vczstore.zarr_impl import append_harmonise

from .utils import (
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
)


@pytest.mark.parametrize(
    (
        "variant_allele",
        "variant_allele_new",
        "expected_variant_allele_updated",
        "expected_variant_allele_new_mapping",
    ),
    [
        # no difference ref only
        ([["A"]], [["A"]], [["A"]], [[0]]),
        # no difference
        ([["A", "C"]], [["A", "C"]], [["A", "C"]], [[0, 1]]),
        # alt differs
        ([["A", "C"]], [["A", "G"]], [["A", "C", "G"]], [[0, 2]]),
        # extra alt
        ([["A", "C"]], [["A", "C", "G"]], [["A", "C", "G"]], [[0, 1, 2]]),
        # extra alt, different order
        ([["A", "C"]], [["A", "G", "C"]], [["A", "C", "G"]], [[0, 2, 1]]),
        # fill values
        (
            [["A", ""], ["A", "C"]],
            [["A", ""], ["A", "G"]],
            [["A", "", ""], ["A", "C", "G"]],
            [[0, -2], [0, 2]],
        ),
        ([["A", "", ""]], [["A", "C", ""]], [["A", "C", ""]], [[0, 1, -2]]),
    ],
)
def test_harmonise_alleles(
    variant_allele,
    variant_allele_new,
    expected_variant_allele_updated,
    expected_variant_allele_new_mapping,
):
    variant_allele_updated, variant_allele_new_mapping = harmonise_alleles(
        np.array(variant_allele), np.array(variant_allele_new)
    )

    assert_array_equal(variant_allele_updated, expected_variant_allele_updated)
    assert_array_equal(variant_allele_new_mapping, expected_variant_allele_new_mapping)


def test_harmonise_alleles__different_num_variants():
    with pytest.raises(
        ValueError,
        match="Different number of variants: store has 2 variants, "
        "but appended data has 1",
    ):
        harmonise_alleles(np.array([["A"], ["C"]]), np.array([["A"]]))


def test_harmonise_alleles__reference_mismatch():
    with pytest.raises(
        ValueError,
        match="References don't match at index 0: store is 'A', "
        "but appended data is 'T'",
    ):
        harmonise_alleles(np.array([["A"]]), np.array([["T"]]))


def test_append_different_alleles(tmp_path):
    vcz1 = convert_vcf_to_vcz("alleles-1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("alleles-2.vcf.gz", tmp_path)

    root1 = zarr.open(vcz1, mode="r")
    root2 = zarr.open(vcz2, mode="r")

    variant_allele1 = root1["variant_allele"][:]
    variant_allele2 = root2["variant_allele"][:]

    variant_allele1_updated, variant_allele2_mapping = harmonise_alleles(
        variant_allele1, variant_allele2
    )

    assert_array_equal(variant_allele1_updated, [["A", "", ""], ["A", "C", "G"]])
    assert_array_equal(variant_allele2_mapping, [[0, -2], [0, 2]])


def test_append_harmonise(tmp_path):
    vcz1 = convert_vcf_to_vcz("alleles-1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("alleles-2.vcf.gz", tmp_path)

    append_harmonise(vcz1, vcz2)

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version",
        "alleles-merged.vcf.gz",
        "view --no-version",
        vcz1,
    )
