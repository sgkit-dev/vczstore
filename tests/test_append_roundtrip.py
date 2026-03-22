import pathlib

import pytest
import zarr
from bio2zarr import vcf
from vcztools.vcf_writer import dims

from vczstore.zarr_impl import append_harmonise

from .utils import (
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    run_bcftools,
)


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "1kg_2020_chrM.vcf.gz",
    ],
)
def test_split_append_roundtrip(tmp_path, vcf_file):
    original = pathlib.Path("tests/data/vcf") / vcf_file

    # trim alt alleles for original
    original_trimmed = tmp_path.joinpath(f"{vcf_file}-trimmed.vcf.gz")
    run_bcftools(
        f"view --no-update {original} -o {original_trimmed} "
        "--write-index=tbi --trim-alt-alleles"
    )

    # convert to VCZ
    vcz_trimmed = (pathlib.Path(tmp_path) / original_trimmed.name).with_suffix(".vcz")
    vcf.convert(
        [original_trimmed], vcz_trimmed, worker_processes=0, local_alleles=False
    )

    # find all the sample IDs in the VCF
    bcftools_out, _ = run_bcftools(f"query -l {original}")
    all_samples = bcftools_out.strip().split("\n")

    # create individual sample VCFs and convert to VCZ
    single_sample_vczs = []
    for sample in all_samples:
        single_sample_vcf = tmp_path.joinpath(f"{vcf_file}-{sample}.vcf.gz")
        run_bcftools(
            f"view -s {sample} --no-update {original} -o {single_sample_vcf} "
            "--write-index=tbi --trim-alt-alleles"
        )
        vcz = convert_vcf_to_vcz(single_sample_vcf, tmp_path)
        single_sample_vczs.append(vcz)

    # append them all together
    store = single_sample_vczs[0]
    for vcz in single_sample_vczs[1:]:
        append_harmonise(store, vcz)

    # check that it has same shapes as the VCZ converted from original
    root_original = zarr.open(vcz_trimmed)
    root_appended = zarr.open(store)
    for var in root_original.keys():
        arr_original = root_original[var]
        arr_appended = root_appended[var]
        assert dims(arr_appended) == dims(arr_original)
        assert arr_appended.shape == arr_original.shape, f"{var} shape differs"

    # check equivalence with original trimmed VCF
    # (note that there is no guarantee that the alt allele ordering is the same,
    # but it happens to be for the VCFs tested here)
    compare_vcf_and_vcz(
        tmp_path, "view --no-version", original_trimmed, "view --no-version", store
    )
