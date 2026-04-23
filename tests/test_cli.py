import click.testing as ct

from vczstore import cli
from vczstore.concurrency import DEFAULT_IO_CONCURRENCY


def test_append_cli_accepts_io_concurrency(monkeypatch):
    seen = {}

    def fake_append(vcz1, vcz2, *, show_progress=False, io_concurrency=None):
        seen["args"] = (vcz1, vcz2)
        seen["show_progress"] = show_progress
        seen["io_concurrency"] = io_concurrency

    monkeypatch.setattr(cli, "append_function", fake_append)

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["append", "--io-concurrency", "4", "--no-progress", "left", "right"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert seen["args"] == ("left", "right")
    assert seen["show_progress"] is False
    assert seen["io_concurrency"] == 4


def test_copy_store_to_icechunk_cli_passes_through_io_concurrency(monkeypatch):
    seen = {}

    def fake_copy_store_to_icechunk(source, dest, *, io_concurrency=None):
        seen["args"] = (source, dest)
        seen["io_concurrency"] = io_concurrency

    monkeypatch.setattr(
        "vczstore.icechunk_utils.copy_store_to_icechunk", fake_copy_store_to_icechunk
    )

    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["copy-store-to-icechunk", "--io-concurrency", "3", "left", "right"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert seen["args"] == ("left", "right")
    assert seen["io_concurrency"] == 3


def test_old_copy_concurrency_flag_is_removed():
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["copy-store-to-icechunk", "--copy-concurrency", "3", "left", "right"],
    )

    assert result.exit_code != 0
    assert "--copy-concurrency" in result.output


def test_help_shows_current_io_concurrency_default():
    runner = ct.CliRunner()
    result = runner.invoke(
        cli.vczstore_main,
        ["append", "--help"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert f"internal default ({DEFAULT_IO_CONCURRENCY})" in result.output
