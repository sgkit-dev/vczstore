import click.testing as ct
import pytest

from vczstore import cli

from .utils import (
    check_removed_sample,
    convert_vcf_to_vcz,
    run_vcztools,
)


@pytest.mark.parametrize(
    ("command", "command_args", "function_name", "expected_args"),
    [
        (
            "normalise",
            ["left", "right", "out"],
            "normalise_function",
            ("left", "right", "out"),
        ),
        ("remove", ["store", "S1"], "remove_function", ("store", "S1")),
    ],
)
@pytest.mark.parametrize(
    ("progress_args", "expected_progress"),
    [
        ([], True),
        (["--no-progress"], False),
    ],
)
@pytest.mark.parametrize(
    ("zarr_backend_storage_args", "expected_zarr_backend_storage"),
    [
        ([], None),
        (["--zarr-backend-storage", "fsspec"], "fsspec"),
        (["--zarr-backend-storage", "icechunk"], "icechunk"),
        (["--zarr-backend-storage", "obstore"], "obstore"),
    ],
)
def test_commands_pass_arguments_and_options(
    monkeypatch,
    command,
    command_args,
    function_name,
    expected_args,
    progress_args,
    expected_progress,
    zarr_backend_storage_args,
    expected_zarr_backend_storage,
):
    seen = {}

    def fake_function(*args, show_progress=False, zarr_backend_storage=None, **kwargs):
        seen["args"] = args
        seen["show_progress"] = show_progress
        seen["zarr_backend_storage"] = zarr_backend_storage

    monkeypatch.setattr(cli, function_name, fake_function)

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        [command, *progress_args, *zarr_backend_storage_args, *command_args],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert seen["args"] == expected_args
    assert seen["show_progress"] is expected_progress
    assert seen["zarr_backend_storage"] == expected_zarr_backend_storage


def test_copy_store_to_icechunk_cli_delegates_to_copy_function(monkeypatch):
    seen = {}

    def fake_copy_store_to_icechunk(source, dest):
        seen["args"] = (source, dest)

    monkeypatch.setattr(
        "vczstore.icechunk_utils.copy_store_to_icechunk", fake_copy_store_to_icechunk
    )

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["copy-store-to-icechunk", "left", "right"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert seen["args"] == ("left", "right")


def test_append_cli_passes_io_concurrency_and_direct_copy_flag(monkeypatch):
    seen = {}

    def fake_append(
        vcz1, vcz2, *, io_concurrency=None, require_direct_copy=False, **kwargs
    ):
        seen["args"] = (vcz1, vcz2)
        seen["io_concurrency"] = io_concurrency
        seen["require_direct_copy"] = require_direct_copy

    monkeypatch.setattr(cli, "append_function", fake_append)

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        [
            "append",
            "--io-concurrency",
            "64",
            "--require-direct-copy",
            "left",
            "right",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert seen == {
        "args": ("left", "right"),
        "io_concurrency": 64,
        "require_direct_copy": True,
    }


def test_help_lists_commands_in_natural_order():
    runner = ct.CliRunner()
    result = runner.invoke(cli.vczstore_main, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert list(cli.vczstore_main.commands) == [
        "append",
        "normalise",
        "remove",
        "copy-store-to-icechunk",
    ]


@pytest.mark.parametrize(
    ("command", "command_args"),
    [
        ("append", ["left", "right"]),
        ("remove", ["store", "S1"]),
    ],
)
def test_invalid_zarr_backend_storage_is_rejected(command, command_args):
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        [command, "--zarr-backend-storage", "bogus", *command_args],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--zarr-backend-storage'" in result.output
    assert "bogus" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["append", "only-one-store"],
        ["normalise", "left", "right"],
        ["remove", "store"],
        ["copy-store-to-icechunk", "only-one-store"],
    ],
)
def test_missing_required_arguments_are_rejected(args):
    runner = ct.CliRunner()
    result = runner.invoke(cli.vczstore_main, args)

    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_cli_reports_operation_value_errors(monkeypatch):
    def fake_append(vcz1, vcz2, **kwargs):
        raise ValueError("stores do not line up")

    monkeypatch.setattr(cli, "append_function", fake_append)

    runner = ct.CliRunner()
    result = runner.invoke(cli.vczstore_main, ["append", "left", "right"])

    assert result.exit_code == 1
    assert "Error: stores do not line up" in result.output


def test_append_cli_updates_vcz_store(tmp_path):
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path, samples_chunk_size=2)
    vcz2 = convert_vcf_to_vcz("sample-part2.vcf.gz", tmp_path)

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["append", str(vcz1), str(vcz2)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    vcztools_out, _ = run_vcztools(f"query -l {vcz1}")
    assert vcztools_out.strip() == "NA00001\nNA00002\nNA00003"


def test_remove_cli_updates_vcz_store(tmp_path):
    vcz = convert_vcf_to_vcz("sample.vcf.gz", tmp_path)

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["remove", "--no-progress", str(vcz), "NA00002"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    vcztools_out, _ = run_vcztools(f"query -l {vcz}")
    assert vcztools_out.strip() == "NA00001\nNA00003"
    check_removed_sample(vcz, "NA00002")


def test_append_cli_reports_real_validation_error(tmp_path):
    vcz1 = convert_vcf_to_vcz("sample-part1.vcf.gz", tmp_path)
    vcz2 = convert_vcf_to_vcz("alleles-1.vcf.gz", tmp_path)

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["append", str(vcz1), str(vcz2)],
    )

    assert result.exit_code == 1
    assert "Error: Stores being appended must have same number of variants" in (
        result.output
    )
