import numpy as np
import pytest
import zarr
from vcztools.utils import open_zarr

from tests.utils import make_vcz
from vczstore.rechunk import compute_variants_chunk_size_from_target, rechunk


def make_simple_vcz(variants_chunk_size=2):
    return make_vcz(
        variant_contig=[0, 0, 0, 0],
        variant_position=[1, 2, 3, 4],
        alleles=[
            ["A", "T"],
            ["A", "C"],
            ["A", "C"],
            ["A", "G"],
        ],
        sample_id=["S1", "S2"],
        call_genotype=[
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 1]],
        ],
        variants_chunk_size=variants_chunk_size,
    )


def test_rechunk():
    vcz = make_simple_vcz()

    rechunk(vcz, "variant_contig", 4)

    root = open_zarr(vcz)
    assert root["variant_contig"].chunks[0] == 4
    assert root["variant_position"].chunks[0] == 2
    assert root["variant_allele"].chunks[0] == 2
    assert root["call_genotype"].chunks[0] == 2


def test_rechunk_no_variants_axis():
    vcz = make_simple_vcz()
    with pytest.raises(ValueError, match="Array 'contig_id' does not have variants"):
        rechunk(vcz, "contig_id", 4)


def test_rechunk_not_multiple_of_min_chunk_size():
    vcz = make_simple_vcz()
    with pytest.raises(ValueError, match="not an exact multiple"):
        rechunk(vcz, "variant_contig", 3)


def test_rechunk_with_target_uncompressed_size():
    vcz = make_simple_vcz()

    rechunk(vcz, "variant_contig", target_uncompressed_size_bytes=16)

    root = open_zarr(vcz)
    # variant_contig is int32 (4 bytes), 1D; bytes_per_variant_chunk=4, n=4,
    # rounded to multiple of 2 → 4
    assert root["variant_contig"].chunks[0] == 4
    assert root["call_genotype"].chunks[0] == 2


@pytest.mark.parametrize(
    ("dtype", "extra_chunks", "target_bytes", "min_chunk_size", "expected"),
    [
        # 1D int32: bytes_per_variant_chunk=4, target=16 → n=4, multiple of 2 → 4
        (np.int32, (), 16, 2, 4),
        # 1D int32: target=20 → n=5, rounded down to multiple of 2 → 4
        (np.int32, (), 20, 2, 4),
        # 1D int32: target too small → clamp to min_chunk_size
        (np.int32, (), 1, 2, 2),
        # 2D int8 with extra chunk dim 3: bytes_per_variant_chunk=3,
        # target=12 → n=4, multiple of 2 → 4
        (np.int8, (3,), 12, 2, 4),
    ],
)
def test_compute_variants_chunk_size_from_target(
    dtype, extra_chunks, target_bytes, min_chunk_size, expected
):
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store)
    shape = (10,) + tuple(c * 2 for c in extra_chunks)
    chunks = (2,) + extra_chunks
    arr = root.create_array(name="x", data=np.zeros(shape, dtype=dtype), chunks=chunks)

    result = compute_variants_chunk_size_from_target(arr, target_bytes, min_chunk_size)

    assert result == expected
    assert result % min_chunk_size == 0
