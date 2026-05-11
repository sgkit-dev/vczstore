import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from vczstore.remove import remove

from .utils import make_vcz


def test_remove():
    vcz = make_vcz(
        variant_contig=[0, 0, 0, 0],
        variant_position=[1, 2, 3, 4],
        alleles=[
            ["A", "T"],
            ["A", "C"],
            ["A", "C"],
            ["A", "T", "C"],
        ],
        sample_id=["S1", "S2", "S3"],
        call_genotype=[
            [[0, 0], [1, 0], [1, 1]],
            [[0, 1], [0, 0], [0, 1]],
            [[0, 0], [1, 1], [0, 1]],
            [[1, 1], [0, 1], [0, 2]],
        ],
    )

    remove(vcz, "S2")

    root = zarr.open(vcz)

    assert_array_equal(root["variant_contig"][:], [0, 0, 0, 0])
    assert_array_equal(root["variant_position"][:], [1, 2, 3, 4])
    assert_array_equal(
        root["variant_allele"][:],
        [
            ["A", "T", ""],
            ["A", "C", ""],
            ["A", "C", ""],
            ["A", "T", "C"],
        ],
    )
    assert_array_equal(root["sample_id"][:], ["S1", "", "S3"])
    assert_array_equal(
        root["call_genotype"][:],
        [
            [[0, 0], [-1, -1], [1, 1]],
            [[0, 1], [-1, -1], [0, 1]],
            [[0, 0], [-1, -1], [0, 1]],
            [[1, 1], [-1, -1], [0, 2]],
        ],
    )


def test_remove_fails_for_misaligned_call_chunks():
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
        chunks=(2, 1, 2),
        dimension_names=["variants", "samples", "ploidy"],
        compressors=None,
        filters=None,
    )
    root.create_array(
        "call_quality",
        data=np.array([[10, 20], [30, 40]], dtype=np.int16),
        chunks=(1, 2),  # different chunk size in variants dimension
        dimension_names=["variants", "samples"],
        compressors=None,
        filters=None,
    )

    with pytest.raises(
        ValueError, match=r"call_\* fields must share a single variants chunk size"
    ):
        remove(vcz, "S1")

    root_after = zarr.open_group(store=vcz, mode="r")
    np.testing.assert_array_equal(root_after["sample_id"][:], np.array(["S1"]))
