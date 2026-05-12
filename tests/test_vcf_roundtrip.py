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

import pytest

from vczstore.append import append
from vczstore.remove import remove

from .utils import (
    check_removed_sample,
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    run_vcztools,
)


@pytest.mark.parametrize("samples_chunk_size", [1, 2, 4])
@pytest.mark.parametrize(
    ("backend_storage", "zarr_format"),
    [(None, None), ("obstore", None), ("icechunk", 3)],
)
def test_append(tmp_path, samples_chunk_size, backend_storage, zarr_format):
    # note that vcz1 is in icechunk, but the dataset being appended, vcz2, needn't be
    vcz1 = convert_vcf_to_vcz(
        "sample-part1.vcf.gz",
        tmp_path,
        samples_chunk_size=samples_chunk_size,
        zarr_format=zarr_format,
        backend_storage=backend_storage,
    )
    vcz2 = convert_vcf_to_vcz("sample-part2.vcf.gz", tmp_path, zarr_format=zarr_format)

    backend_storage_option = (
        "" if backend_storage is None else f"--backend-storage {backend_storage}"
    )

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1} {backend_storage_option}")
    assert vcztools_out.strip() == "NA00001\nNA00002"

    append(vcz1, vcz2, backend_storage=backend_storage)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz1} {backend_storage_option}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version",
        "sample.vcf.gz",
        f"view --no-version {backend_storage_option}",
        vcz1,
    )


def test_append_from_variants_list(tmp_path):
    vcz0 = convert_vcf_to_vcz(
        "sample-variants.vcf.gz", tmp_path, ploidy=2, samples_chunk_size=2
    )
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


@pytest.mark.parametrize(
    ("backend_storage", "zarr_format"),
    [(None, None), ("obstore", None), ("icechunk", 3)],
)
def test_remove(tmp_path, backend_storage, zarr_format):
    vcz = convert_vcf_to_vcz(
        "sample.vcf.gz",
        tmp_path,
        zarr_format=zarr_format,
        backend_storage=backend_storage,
    )

    backend_storage_option = (
        "" if backend_storage is None else f"--backend-storage {backend_storage}"
    )

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz} {backend_storage_option}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    remove(vcz, "NA00002", backend_storage=backend_storage)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz} {backend_storage_option}")
    assert vcztools_out.strip() == "NA00001\nNA00003"

    # check equivalence with original VCF (with sample subsetting)
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version -s NA00001,NA00003 --no-update",
        "sample.vcf.gz",
        f"view --no-version {backend_storage_option}",
        vcz,
    )

    # check sample values are missing
    check_removed_sample(vcz, "NA00002", backend_storage=backend_storage)


def test_remove_multiple_chunks(tmp_path):
    vcz = convert_vcf_to_vcz("chr22.vcf.gz", tmp_path, variants_chunk_size=10)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz}")
    assert len(vcztools_out.strip().split("\n")) == 100

    remove(vcz, "HG00100")

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz}")
    assert "HG00100" not in vcztools_out
    assert len(vcztools_out.strip().split("\n")) == 99

    # check equivalence with original VCF (with sample subsetting)
    reduced_samples = ",".join(vcztools_out.strip().split("\n"))
    compare_vcf_and_vcz(
        tmp_path,
        f"view --no-version -s {reduced_samples} --no-update",
        "chr22.vcf.gz",
        "view --no-version",
        vcz,
    )

    # check sample values are missing
    check_removed_sample(vcz, "HG00100")


@pytest.mark.parametrize(
    ("backend_storage", "zarr_format"),
    [(None, None), ("obstore", None), ("icechunk", 3)],
)
def test_normalise_and_append(tmp_path, backend_storage, zarr_format):
    vcz0 = convert_vcf_to_vcz(
        "sample-variants.vcf.gz",
        tmp_path,
        ploidy=2,
        zarr_format=zarr_format,
        backend_storage=backend_storage,
    )
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path, zarr_format=zarr_format)

    backend_storage_option = (
        "" if backend_storage is None else f"--backend-storage {backend_storage}"
    )

    append(vcz0, vcz1, backend_storage=backend_storage)

    # check equivalence with original VCF
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version",
        "sample-part1.vcf.gz",
        f"view --no-version {backend_storage_option}",
        vcz0,
    )
