import click
import coloredlogs

from vczstore.append import append as append_function
from vczstore.create import create as create_function
from vczstore.normalise import normalise as normalise_function
from vczstore.remove import remove as remove_function


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


def setup_logging(verbosity):
    level = "WARNING"
    if verbosity == 1:
        level = "INFO"
    elif verbosity >= 2:
        level = "DEBUG"
    coloredlogs.install(level=level)


def call_or_error(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


verbose = click.option("-v", "--verbose", count=True, help="Increase verbosity")

progress = click.option(
    "-P /-Q",
    "--progress/--no-progress",
    default=True,
    help="Show progress bars (default: show)",
)

backend_storage = click.option(
    "--backend-storage",
    type=click.Choice(["obstore", "icechunk"]),
    default=None,
    show_default="local if not specified",
    help="Zarr backend storage to use; one of 'obstore' or 'icechunk'.",
)

variant_chunks_in_batch = click.option(
    "--variant-chunks-in-batch",
    type=click.IntRange(min=1),
    default=None,
    show_default="10",
    help="The number of variant chunks to process in each batch",
)

io_concurrency = click.option(
    "--io-concurrency",
    type=click.IntRange(min=1),
    default=None,
    show_default="4 x CPU cores",
    help="Maximum concurrent chunk copy operations.",
)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@verbose
@backend_storage
@io_concurrency
@click.option(
    "--require-direct-copy",
    is_flag=True,
    help=(
        "Fail unless the append can be performed entirely by encoded chunk copy. "
        "This requires a sample chunk-aligned destination and incoming sample count."
    ),
)
def append(vcz1, vcz2, verbose, backend_storage, io_concurrency, require_direct_copy):
    """Append vcz2 to vcz1 in place"""
    setup_logging(verbose)
    call_or_error(
        append_function,
        vcz1,
        vcz2,
        io_concurrency=io_concurrency,
        require_direct_copy=require_direct_copy,
        backend_storage=backend_storage,
    )


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@click.argument("vcz_out", type=click.Path())
@verbose
@progress
@backend_storage
def create(vcz1, vcz2, vcz_out, verbose, progress, backend_storage):
    """Create a new, empty store vcz_out using merged variants from vcz1 and vcz2"""
    setup_logging(verbose)
    call_or_error(
        create_function,
        vcz1,
        vcz2,
        vcz_out,
        show_progress=progress,
        backend_storage=backend_storage,
    )


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@click.argument("vcz2_norm", type=click.Path())
@click.option(
    "--allow-new-alleles",
    is_flag=True,
    help=(
        "If new alleles are found at a variant site in vcz2 the variant_allele array "
        "is updated, otherwise the operation fails if not specified."
    ),
)
@variant_chunks_in_batch
@verbose
@progress
@backend_storage
def normalise(
    vcz1,
    vcz2,
    vcz2_norm,
    allow_new_alleles,
    variant_chunks_in_batch,
    verbose,
    progress,
    backend_storage,
):
    """Normalise variants in vcz2 with respect to vcz1 and write to vcz2_norm"""
    setup_logging(verbose)
    call_or_error(
        normalise_function,
        vcz1,
        vcz2,
        vcz2_norm,
        allow_new_alleles=allow_new_alleles,
        variant_chunks_in_batch=variant_chunks_in_batch,
        show_progress=progress,
        backend_storage=backend_storage,
    )


@click.command()
@click.argument("vcz", type=click.Path())
@click.argument("sample_id", type=str)
@variant_chunks_in_batch
@verbose
@progress
@backend_storage
def remove(vcz, sample_id, variant_chunks_in_batch, verbose, progress, backend_storage):
    """Remove a sample from vcz and overwrite with missing data"""
    setup_logging(verbose)
    call_or_error(
        remove_function,
        vcz,
        sample_id,
        variant_chunks_in_batch=variant_chunks_in_batch,
        show_progress=progress,
        backend_storage=backend_storage,
    )


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@verbose
@io_concurrency
def copy_store_to_icechunk(vcz1, vcz2, verbose, io_concurrency):
    """Copy a Zarr store to a new Icechunk store"""
    from vczstore.utils import (
        copy_store_to_icechunk as copy_store_to_icechunk_function,
    )

    setup_logging(verbose)
    call_or_error(
        copy_store_to_icechunk_function, vcz1, vcz2, io_concurrency=io_concurrency
    )


@click.group(cls=NaturalOrderGroup, name="vczstore")
@click.version_option()
def vczstore_main():
    pass


vczstore_main.add_command(append)
vczstore_main.add_command(create)
vczstore_main.add_command(normalise)
vczstore_main.add_command(remove)
vczstore_main.add_command(copy_store_to_icechunk)
