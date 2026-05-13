import pathlib

from vcztools.utils import open_zarr

from tests.utils import make_vcz
from vczstore.rechunk import rechunk
from vczstore.utils import copy_store_to_icechunk, print_history


def test_rechunk(
    tmpdir,
):
    vcz = make_vcz(
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
        variants_chunk_size=2,
    )

    ic_tmp_path = pathlib.Path(tmpdir) / "icechunk"
    ic_tmp_path.mkdir()
    vcz_ic = pathlib.Path(ic_tmp_path) / "store.vcz"
    copy_store_to_icechunk(vcz, vcz_ic)

    rechunk(vcz_ic, "variant_contig", 4, backend_storage="icechunk")

    print_history(vcz_ic)

    root = open_zarr(vcz_ic, backend_storage="icechunk")
    assert root["variant_contig"].chunks[0] == 4
    assert root["variant_position"].chunks[0] == 2
    assert root["variant_allele"].chunks[0] == 2
    assert root["call_genotype"].chunks[0] == 2

    # check deleted variant is not in root
    assert "variant_contig_delete" not in root
