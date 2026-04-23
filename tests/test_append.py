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

# Create a variants list VCF with no samples.
# Note that the header contains FORMAT fields, even though there are no samples,
# which is necessary for vc2zarr to create empty arrays.

# bin/vcf-drop-samples.sh tests/data/vcf/sample.vcf.gz \
#  tests/data/vcf/sample-variants.vcf.gz


import numpy as np
import pytest
import zarr

from vczstore.append import append

from .utils import (
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    convert_vcf_to_vcz_icechunk,
    run_vcztools,
)


@pytest.mark.parametrize("samples_chunk_size", [1, 2, 4])
@pytest.mark.parametrize("io_concurrency", [1, 4])
def test_append(tmp_path, samples_chunk_size, io_concurrency):
    vcz1 = convert_vcf_to_vcz(
        "sample-part1.vcf.gz", tmp_path, samples_chunk_size=samples_chunk_size
    )
    vcz2 = convert_vcf_to_vcz("sample-part2.vcf.gz", tmp_path)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert vcztools_out.strip() == "NA00001\nNA00002"

    append(vcz1, vcz2, io_concurrency=io_concurrency)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", "sample.vcf.gz", "view --no-version", vcz1
    )


def test_append_from_variants_list(tmp_path):
    vcz0 = convert_vcf_to_vcz("sample-variants.vcf.gz", tmp_path, ploidy=2)
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz0}")
    assert vcztools_out.strip() == ""

    append(vcz0, vcz1)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz0}")
    assert vcztools_out.strip() == "NA00001\nNA00002"

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", "sample-part1.vcf.gz", "view --no-version", vcz0
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


def test_append_fails_for_misaligned_variant_chunks():
    store1 = zarr.storage.MemoryStore()
    root1 = zarr.create_group(store=store1)
    root1.create_array(
        "contig_id",
        data=np.array(["0"]),
        dimension_names=["contigs"],
        compressors=None,
        filters=None,
    )
    root1.create_array(
        "variant_contig",
        data=np.array([0, 0], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root1.create_array(
        "variant_position",
        data=np.array([1, 2], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root1.create_array(
        "variant_allele",
        data=np.array([["A", "T"], ["C", "G"]]),
        chunks=(2, 2),
        dimension_names=["variants", "alleles"],
        compressors=None,
        filters=None,
    )
    root1.create_array(
        "sample_id",
        data=np.array(["S1"]),
        dimension_names=["samples"],
        compressors=None,
        filters=None,
    )
    root1.create_array(
        "call_genotype",
        data=np.array([[[0, 1]], [[1, 1]]], dtype=np.int8),
        chunks=(1, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    store2 = zarr.storage.MemoryStore()
    root2 = zarr.create_group(store=store2)
    for name in ("contig_id", "variant_contig", "variant_position", "variant_allele"):
        root2.create_array(
            name,
            data=root1[name][:],
            chunks=root1[name].chunks,
            dimension_names=root1[name].metadata.dimension_names,
            compressors=None,
            filters=None,
        )
    root2.create_array(
        "sample_id",
        data=np.array(["S2"]),
        dimension_names=["samples"],
        compressors=None,
        filters=None,
    )
    root2.create_array(
        "call_genotype",
        data=np.array([[[0, 0]], [[0, 1]]], dtype=np.int8),
        chunks=(2, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    with pytest.raises(ValueError, match="VCZ-aligned variant chunks"):
        append(store1, store2, io_concurrency=2)

    root1_after = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(root1_after["sample_id"][:], np.array(["S1"]))


def test_append_multiple_chunks(tmp_path):
    vcz1 = convert_vcf_to_vcz(
        "chr22-part1.vcf.gz", tmp_path, variants_chunk_size=10, samples_chunk_size=50
    )
    vcz2 = convert_vcf_to_vcz(
        "chr22-part2.vcf.gz", tmp_path, variants_chunk_size=10, samples_chunk_size=50
    )

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert len(vcztools_out.strip().split("\n")) == 55

    append(vcz1, vcz2)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert len(vcztools_out.strip().split("\n")) == 100

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", "chr22.vcf.gz", "view --no-version", vcz1
    )


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
