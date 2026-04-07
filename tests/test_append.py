# Create the VCF files, one with samples NA00001 and NA00002 and the other with NA00003

# bcftools view -s NA00001,NA00002 --no-update -O z tests/data/vcf/sample.vcf.gz \
#  > tests/data/vcf/sample-part1.vcf.gz
# bcftools view -s NA00003 --no-update -O z tests/data/vcf/sample.vcf.gz \
#  > tests/data/vcf/sample-part2.vcf.gz
# bcftools index -c tests/data/vcf/sample-part1.vcf.gz
# bcftools index -c tests/data/vcf/sample-part2.vcf.gz

# Similarly for chr22.vcf.gz
# bcftools view --no-update \
#  -S <(bcftools query -l tests/data/vcf/chr22.vcf.gz | head -55) \
#  tests/data/vcf/chr22.vcf.gz --write-index=csi -o tests/data/vcf/chr22-part1.vcf.gz
# bcftools view --no-update \
#  -S <(bcftools query -l tests/data/vcf/chr22.vcf.gz | tail -45) \
#  tests/data/vcf/chr22.vcf.gz --write-index=csi -o tests/data/vcf/chr22-part2.vcf.gz


import pytest
import zarr

from vczstore.zarr_impl import append
from vczstore.zarr_partition_impl import append_finalise, append_init, append_partition

from .utils import (
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    convert_vcf_to_vcz_icechunk,
    run_vcztools,
)


@pytest.mark.parametrize("samples_chunk_size", [1, 2, 4])
def test_append(tmp_path, samples_chunk_size):
    vcz1 = convert_vcf_to_vcz(
        "sample-part1.vcf.gz", tmp_path, samples_chunk_size=samples_chunk_size
    )
    vcz2 = convert_vcf_to_vcz("sample-part2.vcf.gz", tmp_path)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert vcztools_out.strip() == "NA00001\nNA00002"

    append(vcz1, vcz2)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", "sample.vcf.gz", "view --no-version", vcz1
    )


def test_append_fail_num_variants_mismatch(tmp_path):
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("alleles-1.vcf.gz", tmp_path)

    with pytest.raises(
        ValueError,
        match="Stores being appended must have same number of variants. "
        "First has 9, second has 2",
    ):
        append(vcz1, vcz2)


def test_append_fail_alleles_mismatch(tmp_path):
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("sample-part2-alleles-mismatch.vcf.gz", tmp_path)

    with pytest.raises(
        ValueError,
        match="Stores being appended must have same values for field 'variant_allele'",
    ):
        append(vcz1, vcz2)


def test_append_icechunk(tmp_path):
    pytest.importorskip("icechunk")
    from vczstore.icechunk_utils import icechunk_transaction

    # note that vcz1 is in icechunk, but the dataset being appended, vcz2, needn't be
    vcz1 = convert_vcf_to_vcz_icechunk("sample-part1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("sample-part2.vcf.gz", tmp_path)

    print(vcz1)
    print(vcz2)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1} --zarr-backend-storage icechunk")
    assert vcztools_out.strip() == "NA00001\nNA00002"

    with icechunk_transaction(vcz1, "main", message="append") as store:
        append(store, vcz2)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1} --zarr-backend-storage icechunk")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version",
        "sample.vcf.gz",
        "view --no-version --zarr-backend-storage icechunk",
        vcz1,
    )


def test_append_partitioned(tmp_path):
    vcz1 = convert_vcf_to_vcz(
        "chr22-part1.vcf.gz", tmp_path, variants_chunk_size=10, samples_chunk_size=50
    )
    vcz2 = convert_vcf_to_vcz(
        "chr22-part2.vcf.gz", tmp_path, variants_chunk_size=10, samples_chunk_size=50
    )

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert len(vcztools_out.strip().split("\n")) == 55

    append_init(vcz1, vcz2, 3)
    append_partition(vcz1, vcz2, 0)
    append_partition(vcz1, vcz2, 1)
    append_partition(vcz1, vcz2, 2)
    append_finalise(vcz1, vcz2)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert len(vcztools_out.strip().split("\n")) == 100

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", "chr22.vcf.gz", "view --no-version", vcz1
    )

    # check append parameters are not in zarr attributes
    root = zarr.open(vcz1, mode="r+")
    assert "append" not in root.attrs
