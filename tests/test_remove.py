import numpy as np
import pytest
import zarr

from vczstore.remove import remove

from .utils import (
    check_removed_sample,
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    convert_vcf_to_vcz_icechunk,
    run_vcztools,
)


@pytest.mark.parametrize("io_concurrency", [1, 4])
def test_remove(tmp_path, io_concurrency):
    vcz = convert_vcf_to_vcz("sample.vcf.gz", tmp_path)

    # check samples query
    vcztools_out, _ = run_vcztools(f"query -l {vcz}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"

    remove(vcz, "NA00002", io_concurrency=io_concurrency)

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

    with repo.transaction("main", message="append") as store:
        remove(store, "NA00002")

    delete_previous_snapshots(repo)

    snapshots = [snapshot for snapshot in repo.ancestry(branch="main")]
    assert len(snapshots) == 2
    # note that 'create' has been deleted
    assert snapshots[0].message == "append"
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
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store)
    root.create_array(
        "variant_contig",
        data=np.array([0, 0], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        "variant_position",
        data=np.array([1, 2], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        "sample_id",
        data=np.array(["S1", "S2"]),
        dimension_names=["samples"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        "call_genotype",
        data=np.array([[[0, 1], [1, 1]], [[0, 0], [0, 1]]], dtype=np.int8),
        chunks=(1, 2, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    with pytest.raises(ValueError, match="VCZ-aligned variant chunks"):
        remove(store, "S1", io_concurrency=2)

    root_after = zarr.open_group(store=store, mode="r")
    np.testing.assert_array_equal(root_after["sample_id"][:], np.array(["S1", "S2"]))
