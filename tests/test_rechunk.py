import pytest
from vcztools.utils import open_zarr

from tests.utils import make_vcz
from vczstore.rechunk import rechunk


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
