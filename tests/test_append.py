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
from zarr.core.sync import sync

from vczstore.append import append

from .utils import (
    compare_vcf_and_vcz,
    convert_vcf_to_vcz,
    convert_vcf_to_vcz_icechunk,
    make_vcz,
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
    vcz1 = make_vcz(
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
    root1 = zarr.open(vcz1, mode="r+")
    root1.create_array(
        "call_genotype",
        data=np.array([[[0, 1]], [[1, 1]]], dtype=np.int8),
        chunks=(1, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    vcz2 = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S2"],
        call_genotype=[[[0, 0]], [[0, 1]]],
        variants_chunk_size=2,
    )

    with pytest.raises(ValueError, match="VCZ-aligned variant chunks"):
        append(vcz1, vcz2)

    root1_after = zarr.open_group(store=vcz1, mode="r")
    np.testing.assert_array_equal(root1_after["sample_id"][:], np.array(["S1"]))


def test_append_fails_for_misaligned_source_variant_chunks():
    vcz1 = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S1"],
        call_genotype=[[[0, 1]], [[1, 1]]],
        variants_chunk_size=2,
    )

    vcz2 = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S2"],
        variants_chunk_size=2,
    )
    # create call_genotype with different variant chunks
    root2 = zarr.open(vcz2, mode="r+")
    root2.create_array(
        "call_genotype",
        data=np.array([[[0, 0]], [[0, 1]]], dtype=np.int8),
        chunks=(1, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )

    with pytest.raises(ValueError, match="VCZ-aligned variant chunks"):
        append(vcz1, vcz2)

    root1_after = zarr.open_group(store=vcz1, mode="r")
    np.testing.assert_array_equal(root1_after["sample_id"][:], np.array(["S1"]))


def test_append_fails_before_mutating_when_source_call_array_is_missing():
    vcz1 = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S1"],
        call_genotype=[[[0, 1]], [[1, 1]]],
        variants_chunk_size=2,
    )

    vcz2 = make_vcz(
        variant_contig=[0, 0],
        variant_position=[1, 2],
        alleles=[
            ["A", "T"],
            ["C", "G"],
        ],
        sample_id=["S2"],
        variants_chunk_size=2,
    )

    with pytest.raises(
        ValueError, match="append requires 'call_genotype' to be present in both stores"
    ):
        append(vcz1, vcz2)

    root1_after = zarr.open_group(store=vcz1, mode="r")
    np.testing.assert_array_equal(root1_after["sample_id"][:], np.array(["S1"]))
    assert root1_after["call_genotype"].shape == (2, 1, 2)


def test_append_rewrites_when_direct_copy_chunks_differ():
    store1 = _create_minimal_append_store(
        ["S1", "S2"],
        _make_genotype(2, 2),
        samples_chunk_size=2,
    )
    store2 = _create_minimal_append_store(
        ["S3", "S4"],
        _make_genotype(2, 2),
        samples_chunk_size=1,
    )

    append(store1, store2)

    root1_after = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(
        root1_after["sample_id"][:], np.array(["S1", "S2", "S3", "S4"])
    )
    np.testing.assert_array_equal(
        root1_after["call_genotype"][:, 2:, :], _make_genotype(2, 2)
    )


def test_append_fails_before_mutating_when_secondary_call_array_is_misaligned():
    store1 = _create_minimal_append_store(
        ["S1", "S2"],
        _make_genotype(2, 2),
        samples_chunk_size=2,
        call_name="call_a",
    )
    store2 = _create_minimal_append_store(
        ["S3", "S4", "S5", "S6"],
        _make_genotype(2, 4),
        samples_chunk_size=2,
        call_name="call_a",
    )
    for store, width in [(store1, 2), (store2, 4)]:
        zarr.open_group(store=store, mode="r+").create_array(
            "call_z",
            data=np.ones((2, width), dtype=np.int8),
            chunks=(2, 4),
            dimension_names=["variants", "samples"],
        )

    append(store1, store2)

    root1_after = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(
        root1_after["sample_id"][:], np.array(["S1", "S2", "S3", "S4", "S5", "S6"])
    )
    np.testing.assert_array_equal(root1_after["call_a"][:, 2:, :], _make_genotype(2, 4))
    np.testing.assert_array_equal(root1_after["call_z"][:, 2:], np.ones((2, 4)))


def test_append_preserves_sparse_source_chunks_as_fill_chunks():
    store1 = _create_minimal_append_store(
        ["S1", "S2"],
        _make_genotype(2, 2),
        samples_chunk_size=2,
    )
    incoming = _make_genotype(2, 4)
    store2 = _create_minimal_append_store(
        ["S3", "S4", "S5", "S6"],
        incoming,
        samples_chunk_size=2,
    )
    source_genotype = zarr.open_group(store=store2, mode="r+")["call_genotype"]
    sync(
        (
            source_genotype.store_path
            / source_genotype.metadata.encode_chunk_key((0, 1, 0))
        ).delete()
    )

    append(store1, store2, io_concurrency=2)

    root = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(root["call_genotype"][:, 2:4, :], incoming[:, :2, :])
    np.testing.assert_array_equal(
        root["call_genotype"][:, 4:6, :],
        np.zeros((2, 2, 2), dtype=np.int8),
    )


def test_append_deletes_stale_destination_chunk_when_source_chunk_is_sparse():
    store1 = _create_minimal_append_store(
        ["S1", "S2"],
        _make_genotype(2, 2),
        samples_chunk_size=2,
    )
    incoming = _make_genotype(2, 4)
    store2 = _create_minimal_append_store(
        ["S3", "S4", "S5", "S6"],
        incoming,
        samples_chunk_size=2,
    )

    dest_genotype = zarr.open_group(store=store1, mode="r+")["call_genotype"]
    source_genotype = zarr.open_group(store=store2, mode="r+")["call_genotype"]
    source_chunk = (
        source_genotype.store_path
        / source_genotype.metadata.encode_chunk_key((0, 1, 0))
    )
    stale_dest_chunk = (
        dest_genotype.store_path / dest_genotype.metadata.encode_chunk_key((0, 2, 0))
    )
    sync(
        stale_dest_chunk.set(
            sync(
                (
                    dest_genotype.store_path
                    / dest_genotype.metadata.encode_chunk_key((0, 0, 0))
                ).get()
            )
        )
    )
    sync(source_chunk.delete())
    assert sync(stale_dest_chunk.get()) is not None

    append(store1, store2, io_concurrency=2)

    assert sync(stale_dest_chunk.get()) is None
    root = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(root["call_genotype"][:, 2:4, :], incoming[:, :2, :])
    np.testing.assert_array_equal(
        root["call_genotype"][:, 4:6, :],
        np.zeros((2, 2, 2), dtype=np.int8),
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


def test_append_icechunk(tmp_path):
    pytest.importorskip("icechunk")
    from vczstore.icechunk_utils import icechunk_transaction

    # note that vcz1 is in icechunk, but the dataset being appended, vcz2, needn't be
    vcz1 = convert_vcf_to_vcz_icechunk("sample-part1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("sample-part2.vcf.gz", tmp_path, zarr_format=3)

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


def _make_genotype(num_variants, num_samples):
    values = np.zeros((num_variants, num_samples, 2), dtype=np.int8)
    for variant_index in range(num_variants):
        for sample_index in range(num_samples):
            values[variant_index, sample_index, 0] = variant_index
            values[variant_index, sample_index, 1] = sample_index
    return values


def _create_minimal_append_store(
    sample_ids, genotype, *, samples_chunk_size, call_name="call_genotype"
):
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store)
    root.create_array(
        "contig_id",
        data=np.array(["0"]),
        dimension_names=["contigs"],
    )
    root.create_array(
        "variant_contig",
        data=np.array([0, 0], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
    )
    root.create_array(
        "variant_position",
        data=np.array([1, 2], dtype=np.int32),
        chunks=(2,),
        dimension_names=["variants"],
    )
    root.create_array(
        "variant_allele",
        data=np.array([["A", "T"], ["C", "G"]]),
        chunks=(2, 2),
        dimension_names=["variants", "alleles"],
    )
    root.create_array(
        "sample_id",
        data=np.array(sample_ids),
        chunks=(samples_chunk_size,),
        dimension_names=["samples"],
    )
    root.create_array(
        call_name,
        data=np.asarray(genotype, dtype=np.int8),
        chunks=(2, samples_chunk_size, 2),
        dimension_names=["variants", "samples", "ploidy"],
    )
    return store


def test_append_preserves_order():
    store1 = _create_minimal_append_store(
        ["S1", "S2"],
        _make_genotype(2, 2),
        samples_chunk_size=2,
    )
    store2 = _create_minimal_append_store(
        ["S3", "S4", "S5", "S6"],
        _make_genotype(2, 4),
        samples_chunk_size=2,
    )

    append(store1, store2, io_concurrency=2)

    root = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(
        root["sample_id"][:], np.array(["S1", "S2", "S3", "S4", "S5", "S6"])
    )
    np.testing.assert_array_equal(root["call_genotype"][:, 2:, :], _make_genotype(2, 4))


def test_append_preserves_order_when_destination_is_not_chunk_aligned():
    store1 = _create_minimal_append_store(
        ["S1", "S2", "S3"],
        _make_genotype(2, 3),
        samples_chunk_size=2,
    )
    incoming = _make_genotype(2, 5)
    store2 = _create_minimal_append_store(
        ["I1", "I2", "I3", "I4", "I5"],
        incoming,
        samples_chunk_size=2,
    )

    append(store1, store2, io_concurrency=2)

    root = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(
        root["sample_id"][:],
        np.array(["S1", "S2", "S3", "I1", "I2", "I3", "I4", "I5"]),
    )
    np.testing.assert_array_equal(root["call_genotype"][:, 3:, :], incoming)


def test_require_direct_copy_fails_before_mutating_when_destination_is_not_aligned():
    store1 = _create_minimal_append_store(
        ["S1", "S2", "S3"],
        _make_genotype(2, 3),
        samples_chunk_size=2,
    )
    store2 = _create_minimal_append_store(
        ["I1", "I2", "I3", "I4"],
        _make_genotype(2, 4),
        samples_chunk_size=2,
    )

    with pytest.raises(ValueError, match="direct-only append"):
        append(store1, store2, require_direct_copy=True)

    root = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(root["sample_id"][:], np.array(["S1", "S2", "S3"]))
    assert root["call_genotype"].shape == (2, 3, 2)


def test_require_direct_copy_fails_before_mutating_when_incoming_is_not_aligned():
    store1 = _create_minimal_append_store(
        ["S1", "S2"],
        _make_genotype(2, 2),
        samples_chunk_size=2,
    )
    store2 = _create_minimal_append_store(
        ["I1", "I2", "I3"],
        _make_genotype(2, 3),
        samples_chunk_size=2,
    )

    with pytest.raises(ValueError, match="direct-only append"):
        append(store1, store2, require_direct_copy=True)

    root = zarr.open_group(store=store1, mode="r")
    np.testing.assert_array_equal(root["sample_id"][:], np.array(["S1", "S2"]))
    assert root["call_genotype"].shape == (2, 2, 2)
