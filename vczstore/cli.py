from contextlib import nullcontext

import click


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


num_partitions = click.option(
    "-n",
    "--num-partitions",
    type=click.IntRange(min=1),
    default=None,
    help="Target number of partitions to use for distributed operations",
)

partition = click.argument("partition", type=click.IntRange(min=0))

impl = click.option(
    "-i",
    "--impl",
    type=str,
    default=None,
    help="Implementation to use; currently only 'zarr' (default).",
)

zarr_backend_storage = click.option(
    "--zarr-backend-storage",
    type=str,
    default=None,
    help="Zarr backend storage to use; one of 'fsspec' (default) or 'icechunk'.",
)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@impl
@zarr_backend_storage
def append(vcz1, vcz2, impl, zarr_backend_storage):
    """Append vcz2 to vcz1 in place"""
    if impl is None or impl == "zarr":
        from vczstore.zarr_impl import append as append_function
    else:
        raise ValueError(f"Unrecognised impl: {impl}")

    if zarr_backend_storage == "icechunk":
        from vczstore.icechunk_utils import icechunk_transaction

        cm = icechunk_transaction(vcz1, "main", message="append")
    else:
        cm = nullcontext(vcz1)
    with cm as vcz1:
        append_function(vcz1, vcz2)


@click.command()
@click.argument("vcz", type=click.Path())
@click.argument("sample_id", type=str)
@impl
@zarr_backend_storage
def remove(vcz, sample_id, impl, zarr_backend_storage):
    """Remove a sample from vcz and overwrite with missing data"""
    if impl is None or impl == "zarr":
        from vczstore.zarr_impl import remove as remove_function
    else:
        raise ValueError(f"Unrecognised impl: {impl}")

    if zarr_backend_storage == "icechunk":
        from vczstore.icechunk_utils import icechunk_transaction

        cm = icechunk_transaction(vcz, "main", message="remove")
    else:
        cm = nullcontext(vcz)
    with cm as vcz:
        remove_function(vcz, sample_id)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@num_partitions
def dappend_init(vcz1, vcz2, num_partitions):
    """
    Initial step for distributed append of vcz2 to vcz1 in place.
    """
    from vczstore.zarr_partition_impl import append_init

    append_init(vcz1, vcz2, num_partitions)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
@partition
def dappend_partition(vcz1, vcz2, partition):
    """
    Append vcz2 to vcz1 in place for a partition.

    Must be called after the distributed append operation has been
    initialised with dappend_init.
    """
    from vczstore.zarr_partition_impl import append_partition

    append_partition(vcz1, vcz2, partition)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
def dappend_finalise(vcz1, vcz2):
    """
    Final step for distributed append of vcz2 to vcz1 in place.
    """
    from vczstore.zarr_partition_impl import append_finalise

    append_finalise(vcz1, vcz2)


@click.command()
@click.argument("vcz", type=click.Path())
@click.argument("sample_id", type=str)
@num_partitions
def dremove_init(vcz, sample_id, num_partitions):
    """
    Initial step for distributed remove of a sample from vcz.
    """
    from vczstore.zarr_partition_impl import remove_init

    remove_init(vcz, sample_id, num_partitions)


@click.command()
@click.argument("vcz", type=click.Path())
@partition
def dremove_partition(vcz, partition):
    """
    Remove a sample from vcz for a partition.

    Must be called after the distributed remove operation has been
    initialised with dremove_init.
    """
    from vczstore.zarr_partition_impl import remove_partition

    remove_partition(vcz, partition)


@click.command()
@click.argument("vcz", type=click.Path())
def dremove_finalise(vcz):
    """
    Final step for distributed remove of a sample from vcz.
    """
    from vczstore.zarr_partition_impl import remove_finalise

    remove_finalise(vcz)


@click.command()
@click.argument("vcz1", type=click.Path())
@click.argument("vcz2", type=click.Path())
def copy_store_to_icechunk(vcz1, vcz2):
    """Copy a Zarr store to a new Icechunk store"""
    from vczstore.icechunk_utils import (
        copy_store_to_icechunk as copy_store_to_icechunk_function,
    )

    copy_store_to_icechunk_function(vcz1, vcz2)


@click.group(cls=NaturalOrderGroup, name="vczstore")
def vczstore_main():
    pass


vczstore_main.add_command(append)
vczstore_main.add_command(remove)
vczstore_main.add_command(copy_store_to_icechunk)
vczstore_main.add_command(dappend_init)
vczstore_main.add_command(dappend_partition)
vczstore_main.add_command(dappend_finalise)
vczstore_main.add_command(dremove_init)
vczstore_main.add_command(dremove_partition)
vczstore_main.add_command(dremove_finalise)
