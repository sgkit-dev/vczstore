#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars>=1.40.1",
# ]
# ///

import sys

import polars as pl


def main(csv_path) -> None:
    df = pl.read_csv(csv_path)
    df = (
        pl.scan_csv(csv_path)
        .with_columns(
            pl.col("cpra_grch38")
            .str.splitn(":", 4)
            .struct.rename_fields(["chrom", "pos", "ref", "alt"])
        )
        .unnest("cpra_grch38")
        .with_columns(
            pl.col("chrom").cast(pl.Categorical),
            pl.col("pos").cast(pl.Int32),
            pl.lit(".").alias("id"),
            pl.lit(".").alias("qual"),
            pl.lit("PASS").alias("filter"),
            pl.lit(".").alias("info"),
        )
        .select(["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"])
        .collect()
    )

    contigs = df.filter(
        (pl.struct("chrom") != pl.struct("chrom").shift(1)).fill_null(True)
    )["chrom"]
    print(
        """##fileformat=VCFv4.3
##FILTER=<ID=PASS,Description="All filters passed">""",
        flush=True,
    )
    for contig in contigs:
        print(f"##contig=<ID={contig}>", flush=True)
    print("#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO", flush=True)

    df.write_csv(sys.stdout.buffer, include_header=False, separator="\t")


if __name__ == "__main__":
    csv_path = sys.argv[1]
    main(csv_path)
