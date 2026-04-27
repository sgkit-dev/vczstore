import logging

import numpy as np
import zarr

from vczstore.utils import variant_chunk_slices, variants_progress

logger = logging.getLogger(__name__)


def append(vcz1, vcz2, *, show_progress=False):
    """Append vcz2 to vcz1 in place"""
    root1 = zarr.open(vcz1, mode="r+")
    root2 = zarr.open(vcz2, mode="r")

    # check preconditions
    n_variants1 = root1["variant_contig"].shape[0]
    n_variants2 = root2["variant_contig"].shape[0]
    if n_variants1 != n_variants2:
        raise ValueError(
            "Stores being appended must have same number of variants. "
            f"First has {n_variants1}, second has {n_variants2}"
        )
    for field in ("contig_id", "variant_contig", "variant_position", "variant_allele"):
        values1 = root1[field][:]
        values2 = root2[field][:]
        if np.any(values1 != values2):
            raise ValueError(
                f"Stores being appended must have same values for field '{field}'"
            )

    # append samples
    sample_id1 = root1["sample_id"]
    sample_id2 = root2["sample_id"]

    old_num_samples = sample_id1.shape[0]
    new_num_samples = old_num_samples + sample_id2.shape[0]
    new_shape = (new_num_samples,)
    sample_id1.resize(new_shape)
    sample_id1[old_num_samples:new_num_samples] = sample_id2[:]

    # TODO: need to check all chunk sizes rather than just last ones!
    old_arr_ends_on_chunk_boundary = False
    new_arr_ends_on_chunk_boundary = False
    sample_chunk_offset = 0

    # resize genotype fields
    for var in root1.keys():
        if var.startswith("call_"):
            arr = root1[var]
            if arr.ndim == 2:
                new_shape = (arr.shape[0], new_num_samples)
                arr.resize(new_shape)
            elif arr.ndim == 3:
                new_shape = (arr.shape[0], new_num_samples, arr.shape[2])
                print(
                    f"resizing {var} from {arr.shape} to {new_shape}, "
                    f"chunks {arr.chunks}"
                )
                sample_chunksize = arr.chunks[1]
                chunks_identical = arr.chunks == root2[var].chunks
                old_arr_ends_on_chunk_boundary = old_num_samples % sample_chunksize == 0
                new_arr_ends_on_chunk_boundary = new_num_samples % sample_chunksize == 0
                sample_chunk_offset = old_num_samples // sample_chunksize
                arr.resize(new_shape)
            else:
                raise ValueError("unsupported number of array_dims")

    print(f"{chunks_identical=}")
    print(f"{old_arr_ends_on_chunk_boundary=}")
    print(f"{new_arr_ends_on_chunk_boundary=}")

    # append genotype fields

    # TODO: check compression is identical (anything else? order, filters)
    if chunks_identical and old_arr_ends_on_chunk_boundary:
        # copy call_genotype here
        from .icechunk_utils import copy_store_chunks

        copy_store_chunks(
            vcz2,
            vcz1,
            array_key="call_genotype",
            chunk_offset=(0, sample_chunk_offset, 0),
        )
    with variants_progress(n_variants1, "Append", show_progress) as pbar:
        for v_sel in variant_chunk_slices(root1):
            for var in root1.keys():
                if var.startswith("call_"):
                    if (
                        chunks_identical
                        and old_arr_ends_on_chunk_boundary
                        and var == "call_genotype"
                    ):
                        continue
                    root1[var][v_sel, old_num_samples:new_num_samples, ...] = root2[
                        var
                    ][v_sel, ...]
            pbar.update(v_sel.stop - v_sel.start)
