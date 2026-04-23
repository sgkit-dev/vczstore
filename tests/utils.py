# Most of this is from vcztools

import os
import pathlib
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import zip_longest

import click.testing as ct
import cyvcf2
import numpy as np
import vcztools.cli as cli
import zarr
from bio2zarr import vcf
from vcztools.utils import array_dims

from vczstore.utils import missing_val


@contextmanager
def open_vcf(path) -> Iterator[cyvcf2.VCF]:
    """A context manager for opening a VCF file."""
    vcf = cyvcf2.VCF(path)
    try:
        yield vcf
    finally:
        vcf.close()


def normalise_info_missingness(info_dict, key):
    value = info_dict.get(key, None)
    if isinstance(value, tuple):
        if all(x is None for x in value):
            value = None
    elif isinstance(value, str):
        if all(x == "." for x in value.split(",")):
            value = None
    return value


def _get_header_field_dicts(vcf, header_type):
    def to_dict(header_field):
        d = header_field.info(extra=True)
        del d[b"IDX"]  # remove IDX since we don't care about ordering

        # cyvcf2 duplicates some keys as strings and bytes, so remove the bytes one
        for k in list(d.keys()):
            if isinstance(k, bytes) and k.decode("utf-8") in d:
                del d[k]
        return d

    return {
        field["ID"]: to_dict(field)
        for field in vcf.header_iter()
        if field["HeaderType"] == header_type
    }


def _assert_header_field_dicts_equivalent(field_dicts1, field_dicts2):
    assert len(field_dicts1) == len(field_dicts2)

    for id in field_dicts1.keys():
        assert id in field_dicts2
        field_dict1 = field_dicts1[id]
        field_dict2 = field_dicts2[id]

        assert len(field_dict1) == len(field_dict2)
        # all fields should be the same, except Number="." which can match any value
        for k in field_dict1.keys():
            assert k in field_dict2
            v1 = field_dict1[k]
            v2 = field_dict2[k]
            if k == "Number" and (v1 == "." or v2 == "."):
                continue
            assert v1 == v2, f"Failed in field {id} with key {k}"


def _assert_vcf_headers_equivalent(vcf1, vcf2):
    # Only compare INFO, FORMAT, FILTER, CONTIG fields, ignoring order
    # Other fields are ignored

    info1 = _get_header_field_dicts(vcf1, "INFO")
    info2 = _get_header_field_dicts(vcf2, "INFO")
    _assert_header_field_dicts_equivalent(info1, info2)

    format1 = _get_header_field_dicts(vcf1, "FORMAT")
    format2 = _get_header_field_dicts(vcf2, "FORMAT")
    _assert_header_field_dicts_equivalent(format1, format2)

    filter1 = _get_header_field_dicts(vcf1, "FILTER")
    filter2 = _get_header_field_dicts(vcf2, "FILTER")
    _assert_header_field_dicts_equivalent(filter1, filter2)

    contig1 = _get_header_field_dicts(vcf1, "CONTIG")
    contig2 = _get_header_field_dicts(vcf2, "CONTIG")
    _assert_header_field_dicts_equivalent(contig1, contig2)


def assert_vcfs_close(f1, f2, *, rtol=1e-05, atol=1e-03, allow_zero_variants=False):
    """Like :py:func:`numpy.testing.assert_allclose()`, but for VCF files.

    Raises an `AssertionError` if two VCF files are not equal to one another.
    Float values in QUAL, INFO, or FORMAT fields are compared up to the
    desired tolerance. All other values must match exactly.

    Parameters
    ----------
    f1
        Path to first VCF to compare.
    f2
        Path to second VCF to compare.
    rtol
        Relative tolerance.
    atol
        Absolute tolerance.
    """
    with open_vcf(f1) as vcf1, open_vcf(f2) as vcf2:
        _assert_vcf_headers_equivalent(vcf1, vcf2)
        assert vcf1.samples == vcf2.samples

        count = 0
        for v1, v2 in zip_longest(vcf1, vcf2):
            if v1 is None and v2 is not None:
                raise AssertionError(f"Right contains extra variant: {v2}")
            if v1 is not None and v2 is None:
                raise AssertionError(f"Left contains extra variant: {v1}")

            count += 1

            assert v1.CHROM == v2.CHROM, f"CHROM not equal for variants\n{v1}{v2}"
            assert v1.POS == v2.POS, f"POS not equal for variants\n{v1}{v2}"
            assert v1.ID == v2.ID, f"ID not equal for variants\n{v1}{v2}"
            assert v1.REF == v2.REF, f"REF not equal for variants\n{v1}{v2}"
            assert v1.ALT == v2.ALT, f"ALT not equal for variants\n{v1}{v2}"
            np.testing.assert_allclose(
                np.array(v1.QUAL, dtype=np.float32),
                np.array(v2.QUAL, dtype=np.float32),
                rtol=rtol,
                atol=atol,
                err_msg=f"QUAL not equal for variants\n{v1}{v2}",
            )
            assert set(v1.FILTERS) == set(v2.FILTERS), (
                f"FILTER not equal for variants\n{v1}{v2}"
            )

            v1_info = dict(v1.INFO)
            v2_info = dict(v2.INFO)
            all_keys = set(v1_info.keys()) | set(v2_info.keys())
            for k in all_keys:
                val1 = normalise_info_missingness(v1_info, k)
                val2 = normalise_info_missingness(v2_info, k)
                # values are python objects (not np arrays)
                if isinstance(val1, float) or (
                    isinstance(val1, tuple) and any(isinstance(v, float) for v in val1)
                ):
                    np.testing.assert_allclose(
                        np.array(val1, dtype=np.float32),
                        np.array(val2, dtype=np.float32),
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"INFO {k} not equal for variants\n{v1}{v2}",
                    )
                else:
                    assert val1 == val2, f"INFO {k} not equal for variants\n{v1}{v2}"

            # NOTE skipping this because it requires items to be in the same order.
            # assert v1.FORMAT == v2.FORMAT, f"FORMAT not equal for variants\n{v1}{v2}"
            for field in v1.FORMAT:
                if field == "GT":
                    assert v1.genotypes == v2.genotypes, (
                        f"GT not equal for variants\n{v1}{v2}"
                    )
                else:
                    val1 = v1.format(field)
                    val2 = v2.format(field)
                    if val2 is None:
                        # FIXME this is a quick hack to workaround missing support for
                        # dealing with the field missing vs all-elements-in-field
                        # missing issue.
                        # https://github.com/jeromekelleher/vcztools/issues/14
                        assert [str(x) == "." for x in val1]
                    else:
                        if val1.dtype.kind == "f":
                            np.testing.assert_allclose(
                                val1,
                                val2,
                                rtol=rtol,
                                atol=atol,
                                err_msg=f"FORMAT {field} not equal for "
                                f"variants\n{v1}{v2}",
                            )
                        else:
                            np.testing.assert_array_equal(
                                val1,
                                val2,
                                err_msg=f"FORMAT {field} not equal for "
                                f"variants\n{v1}{v2}",
                            )

        if not allow_zero_variants:
            assert count > 0, "No variants in file"


def vcz_path_cache(vcf_path):
    """
    Store converted files in a cache to speed up tests. We're not testing
    vcf2zarr here, so no point in running over and over again.
    """
    cache_path = pathlib.Path("vcz_test_cache")
    if not cache_path.exists():
        cache_path.mkdir()
    cached_vcz_path = (cache_path / vcf_path.name).with_suffix(".vcz")
    if not cached_vcz_path.exists():
        if vcf_path.name.startswith("chr22"):
            vcf.convert(
                [vcf_path],
                cached_vcz_path,
                worker_processes=0,
                variants_chunk_size=10,
                samples_chunk_size=10,
            )
        else:
            vcf.convert(
                [vcf_path], cached_vcz_path, worker_processes=0, local_alleles=False
            )
    return cached_vcz_path


def convert_vcf_to_vcz(
    vcf_name, tmpdir, variants_chunk_size=None, samples_chunk_size=None, ploidy=None
):
    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    output = (pathlib.Path(tmpdir) / vcf_path.name).with_suffix(".vcz")
    vcf.convert(
        [vcf_path],
        output,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        worker_processes=0,
        local_alleles=False,
        ploidy=ploidy,
    )
    return output


def run_bcftools(args: str, expect_error=False) -> tuple[str, str]:
    """
    Run bcftools (which must be on the PATH) and return stdout and stderr
    as a pair of strings.
    """
    completed = subprocess.run(
        f"bcftools {args}", capture_output=True, check=False, shell=True
    )
    if expect_error:
        assert completed.returncode != 0
    else:
        assert completed.returncode == 0
    return completed.stdout.decode("utf-8"), completed.stderr.decode("utf-8")


def run_vcztools(args: str, expect_error=False) -> tuple[str, str]:
    """Run run_vcztools and return stdout and stderr as a pair of strings."""
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vcztools_main,
        args,
        catch_exceptions=False,
    )
    if expect_error:
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0
    return result.stdout, result.stderr


def compare_vcf_and_vcz(tmp_path, vcf_args, vcf_file, vcz_args, vcz):
    original = pathlib.Path("tests/data/vcf") / vcf_file

    bcftools_out, _ = run_bcftools(f"{vcf_args} {original}")
    bcftools_out_file = tmp_path.joinpath("bcftools_out.vcf")
    with open(bcftools_out_file, "w") as f:
        f.write(bcftools_out)

    vcztools_out, _ = run_vcztools(f"{vcz_args} {vcz}")
    vcztools_out_file = tmp_path.joinpath("vcztools_out.vcf")
    with open(vcztools_out_file, "w") as f:
        f.write(vcztools_out)

    assert_vcfs_close(bcftools_out_file, vcztools_out_file)


def convert_vcf_to_vcz_icechunk(
    vcf_name, tmp_path, variants_chunk_size=None, samples_chunk_size=None, ploidy=None
):
    from vczstore.icechunk_utils import copy_store_to_icechunk

    vcf_path = pathlib.Path("tests/data/vcf") / vcf_name
    vcz = (pathlib.Path(tmp_path) / vcf_path.name).with_suffix(".vcz")
    env = dict(os.environ)
    env["BIO2ZARR_ZARR_FORMAT"] = "3"
    script = """
from pathlib import Path
import sys

from bio2zarr import vcf

vcf_path = Path(sys.argv[1])
output = Path(sys.argv[2])
variants_chunk_size = None if sys.argv[3] == "None" else int(sys.argv[3])
samples_chunk_size = None if sys.argv[4] == "None" else int(sys.argv[4])
ploidy = None if sys.argv[5] == "None" else int(sys.argv[5])

vcf.convert(
    [vcf_path],
    output,
    variants_chunk_size=variants_chunk_size,
    samples_chunk_size=samples_chunk_size,
    worker_processes=0,
    local_alleles=False,
    ploidy=ploidy,
)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(vcf_path),
            str(vcz),
            str(variants_chunk_size),
            str(samples_chunk_size),
            str(ploidy),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    ic_tmp_path = tmp_path / "icechunk"
    ic_tmp_path.mkdir()
    output = (pathlib.Path(ic_tmp_path) / vcf_name).with_suffix(".vcz")

    copy_store_to_icechunk(vcz, output)

    return output


def check_removed_sample(vcz, sample_id):
    root = zarr.open(vcz)
    all_samples = root["sample_id"][:]

    # check removed sample is not present in sample_id array
    assert sample_id not in all_samples

    # find indexes of removed samples
    selection = all_samples == ""

    # and check all genotype arrays have missing values for those samples
    for var in root.keys():
        arr = root[var]
        if (
            var.startswith("call_")
            and array_dims(arr)[0] == "variants"
            and array_dims(arr)[1] == "samples"
        ):
            np.testing.assert_array_equal(
                root[var][:][:, selection, ...], missing_val(arr)
            )
