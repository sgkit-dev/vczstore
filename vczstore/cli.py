from contextlib import nullcontext

import click

from vczstore.append import append as append_function
from vczstore.normalise import normalise as normalise_function
from vczstore.remove import remove as remove_function


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


def show_work_summary(work_summary):
    output = work_summary.asjson()
    click.echo(output)


def call_or_error(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


progress = click.option(
    "-P /-Q",
    "--progress/--no-progress",
    default=True,
    help="Show progress bars (default: show)",
)


zarr_backend_storage = click.option(
    "--zarr-backend-storage",
    type=click.Choice(["fsspec", "obstore", "icechunk"]),
    default=None,
    help="Zarr backend storage to use; one of 'fsspec' (default), 'obstore', "
    "or 'icechunk'.",
)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@zarr_backend_storage
@click.option(
    "--io-concurrency",
    type=click.IntRange(min=1),
    default=None,
    show_default="4 x CPU cores",
    help="Maximum concurrent chunk copy operations.",
)
@click.option(
    "--require-direct-copy",
    is_flag=True,
    help=(
        "Fail unless the append can be performed entirely by encoded chunk copy. "
        "This requires a sample chunk-aligned destination and incoming sample count."
    ),
)
def append(vcz1, vcz2, zarr_backend_storage, io_concurrency, require_direct_copy):
    """Append vcz2 to vcz1 in place"""
    if zarr_backend_storage == "icechunk":
        from vczstore.icechunk_utils import icechunk_transaction

        cm = icechunk_transaction(vcz1, "main", message="append")
    else:
        cm = nullcontext(vcz1)
    with cm as vcz1:
        call_or_error(
            append_function,
            vcz1,
            vcz2,
            io_concurrency=io_concurrency,
            require_direct_copy=require_direct_copy,
            zarr_backend_storage=zarr_backend_storage,
        )


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@click.argument("vcz2_norm", type=click.Path())
@progress
def normalise(vcz1, vcz2, vcz2_norm, progress):
    """Normalise variants in vcz2 with respect to vcz1 and write to vcz2_norm"""
    call_or_error(
        normalise_function,
        vcz1,
        vcz2,
        vcz2_norm,
        show_progress=progress,
        zarr_backend_storage=zarr_backend_storage,
    )


@click.command()
@click.argument("vcz", type=click.Path())
@click.argument("sample_id", type=str)
@progress
@zarr_backend_storage
def remove(vcz, sample_id, progress, zarr_backend_storage):
    """Remove a sample from vcz and overwrite with missing data"""
    if zarr_backend_storage == "icechunk":
        from vczstore.icechunk_utils import icechunk_transaction

        cm = icechunk_transaction(vcz, "main", message="remove")
    else:
        cm = nullcontext(vcz)
    with cm as vcz:
        call_or_error(
            remove_function,
            vcz,
            sample_id,
            show_progress=progress,
            zarr_backend_storage=zarr_backend_storage,
        )


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
def copy_store_to_icechunk(vcz1, vcz2):
    """Copy a Zarr store to a new Icechunk store"""
    from vczstore.icechunk_utils import (
        copy_store_to_icechunk as copy_store_to_icechunk_function,
    )

    call_or_error(copy_store_to_icechunk_function, vcz1, vcz2)


@click.group(cls=NaturalOrderGroup, name="vczstore")
def vczstore_main():
    pass


vczstore_main.add_command(append)
vczstore_main.add_command(normalise)
vczstore_main.add_command(remove)
vczstore_main.add_command(copy_store_to_icechunk)
