import numpy as np
import pytest
import zarr

from vczstore.remove import remove

from .utils import (
    check_removed_sample,
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    convert_vcf_to_vcz_icechunk,
    make_vcz,
    run_vcztools,
)


@pytest.mark.parametrize("zarr_backend_storage", [None, "obstore", "fsspec"])
def test_remove(tmp_path, zarr_backend_storage):
    vcz = convert_vcf_to_vcz("sample.vcf.gz", tmp_path)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    remove(vcz, "NA00002", zarr_backend_storage=zarr_backend_storage)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz}")
    assert vcztools_out.strip() == "NA00001\nNA00003"

    # check equivalence with original VCF (with sample subsetting)
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version -s NA00001,NA00003 --no-update",
        "sample.vcf.gz",
        "view --no-version",
        vcz,
    )

    # check sample values are missing
    check_removed_sample(vcz, "NA00002")


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


def test_remove_icechunk(tmp_path):
    pytest.importorskip("icechunk")
    from icechunk import Repository

    from vczstore.icechunk_utils import delete_previous_snapshots, make_icechunk_storage

    vcz = convert_vcf_to_vcz_icechunk("sample.vcf.gz", tmp_path)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz} --zarr-backend-storage icechunk")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    icechunk_storage = make_icechunk_storage(vcz)
    repo = Repository.open(icechunk_storage)

    snapshots = [snapshot for snapshot in repo.ancestry(branch="main")]
    assert len(snapshots) == 2
    assert snapshots[0].message == "create"
    assert snapshots[1].message == "Repository initialized"

    remove(vcz, "NA00002", zarr_backend_storage="icechunk")

    delete_previous_snapshots(repo)

    snapshots = [snapshot for snapshot in repo.ancestry(branch="main")]
    assert len(snapshots) == 2
    # note that 'create' has been deleted
    assert snapshots[0].message == "remove"
    assert snapshots[1].message == "Repository initialized"

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz} --zarr-backend-storage icechunk")
    assert vcztools_out.strip() == "NA00001\nNA00003"

    # check equivalence with original VCF (with sample subsetting)
    compare_vcf_and_vcz(
        tmp_path,
        "view --no-version -s NA00001,NA00003 --no-update",
        "sample.vcf.gz",
        "view --no-version --zarr-backend-storage icechunk",
        vcz,
    )

    # check sample values are missing
    session = repo.readonly_session("main")
    store = session.store
    check_removed_sample(store, "NA00002")


def test_remove_fails_for_misaligned_variant_chunks():
    vcz = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S1"],
        variants_chunk_size=2,
    )
    # create call_genotype with different variant chunks
    root = zarr.open(vcz, mode="r+")
    root.create_array(
        "call_genotype",
        data=np.array([[[0, 1]], [[1, 1]]], dtype=np.int8),
        chunks=(1, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    with pytest.raises(ValueError, match="VCZ-aligned variant chunks"):
        remove(vcz, "S1")

    root_after = zarr.open_group(store=vcz, mode="r")
    np.testing.assert_array_equal(root_after["sample_id"][:], np.array(["S1"]))


def test_remove_fails_for_malformed_call_array_dimensions():
    vcz = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S1"],
        variants_chunk_size=2,
    )
    # create call_quality with invalid dimensions
    root = zarr.open(vcz, mode="r+")
    root.create_array(
        "call_quality",
        data=np.array([[10, 20], [30, 40]], dtype=np.int16),
        chunks=(2, 2),
        dimension_names=["variants", "not_samples"],
        compressors=None,
        filters=None,
    )

    with pytest.raises(
        ValueError,
        match="remove requires 'call_quality' to use variants/samples dimensions",
    ):
        remove(vcz, "S1")

    root_after = zarr.open_group(store=vcz, mode="r")
    np.testing.assert_array_equal(root_after["sample_id"][:], np.array(["S1"]))
