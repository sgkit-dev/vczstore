# vczstore

Tools for managing a VCF Zarr store.

## Introduction

The [VCF Zarr format](https://github.com/sgkit-dev/vcf-zarr-spec/) enables efficient, scalable storage and analysis of large-scale genetic variation data.

VCF Zarr files (also known as VCZ files) can be created using [bio2zarr](https://sgkit-dev.github.io/bio2zarr/intro.html). While ideal for one-off conversion of VCF files to Zarr, bio2zarr does not support updating a VCF Zarr with new samples.

Vczstore solves the update use case by providing the following operations

1. **Create** an empty VCZ store with a known set of variants
2. **Append** new samples to a VCZ store
3. **Remove** samples from a VCZ store

The append operation works by appending a VCF Zarr file to the store (which is another VCF Zarr). The file being appended is created using bio2zarr.

## Variant sets

Note that only new _samples_ can be added - not new variants - so the store contains a fixed set of variants that must be known before it is created. This is usually not a limitation since the samples come from the same genotype array, or even from the same reference panel for imputed data.

If the source VCFs have different (but typically overlapping) sets of variants, then they need to be harmonised with the full set of variants before being converted to VCZ. This can be accomplished by running `vczstore normalise` before running append.

The append operation will perform a check that the contig, position and allele (REF and ALT) fields all match before performing the update. It will fail if there is a mismatch, so samples with inconsistent variant sets cannot be appended. The check is strict - allele ordering must match exactly too.

In rare cases, samples may have new alleles at a site. This is permitted and as long as the `--allow-new-alleles` option is used with `vczstore normalise`, the new alleles will be added to the `variant_allele` array when `vczstore append` is used to append the new samples.

Multiallelic sites, and split alleles (mutiple records for a site) are both accepted, as long as the ordering is consistent for all source VCZs. 

## Operations

### Creating a store

A VCZ store is just a VCZ file - typically in cloud object store - so it's possible to create one using bio2zarr (e.g. vcf2zarr).

However, when the VCFs being appended contain different variants the store must be created with the full set of variants. This is achieved by calling `vczstore create` with the VCZ files that collectively define the set of variants (e.g. one for each genotype array). (Currently `vczstore create` can only be called with two arguments - but you can call it repeatedly to built up the store from multiple files.)

After creation the store contains no samples.

![Creating a store](docs/images/vczstore-create.drawio.svg)

### Appending new samples to a store

New samples often arrive as single-sample VCFs. For efficiency, it is best to append in batches that correspond to the sample chunk size (default 10,000), but it is possible to append a smaller number of samples at a significant performance cost (typically an order of magnitude slower). There is a CLI option, `--require-direct-copy`, that will cause the append to fail before performing the update unless the chunks align to allow a direct copy.

The single-sample VCFs are turned into a single VCF using `bcftools merge`. However, note that in this case the samples must all have the same set of variants. If there are VCFs with different variants (e.g. from different genotype arrays) then they must be appended in separate operations.

The `vczstore normalise` step ensures that the VCZ contains the same number of variants as the store (and in the same order), so the new sample data can simply be appended to the end of the genotype arrays.

![Appending new samples to a store](docs/images/vczstore-append.drawio.svg)

### Removing a sample from a store

When a sample is removed, all data for that sample is overwritten with missing values. (Note that this means that storage space is not reclaimed.) This is performed directly on the store.

When using Icechunk storage, the previous commit is [amended](https://icechunk.io/en/stable/understanding/version-control/#amending-a-snapshot), which ensures that no history for removed samples is retained in the store.

![Removing a sample from a store](docs/images/vczstore-remove.drawio.svg)

## Implementation

The implementation uses zarr-python (version 3) directly to update Zarr chunks in the store. Stores using Zarr format 2 and 3 are supported, as well as Icechunk storage. Using Icechunk means that updates are performed in a transaction so that other users accessing the store will see consistent updates.

| zarr-python (format) | v2                 | v3                 |
|----------------------|--------------------|--------------------|
| **no transactions**  | :white_check_mark: | :white_check_mark: |
| **icechunk**         | N/A                | :white_check_mark: |

VCF Zarr stores can reside in cloud stores such as Amazon S3 or Azure Cloud Storage.
For Icechunk-backed stores, local filesystem paths plus `s3://`, `az://`,
`azure://`, `abfs://`, `abfss://`, and Azure
`https://...blob.core.windows.net/...` or `https://...dfs.core.windows.net/...`
URLs are supported.

## Demo

* Transactions: none
* Distributed: single process
* Zarr: format 2

```shell
% uv sync --group dev

# Create some VCZ data
% rm -rf data
% mkdir data
% uv run vcf2zarr convert --no-progress --samples-chunk-size=4 tests/data/vcf/sample-part1.vcf.gz data/store.vcz
% uv run vcf2zarr convert --no-progress tests/data/vcf/sample-part2.vcf.gz data/sample-part2.vcf.vcz

# Show the samples in each
% uv run vcztools query -l data/store.vcz
NA00001
NA00002
% uv run vcztools query -l data/sample-part2.vcf.vcz
NA00003

# Append data to the store
% uv run vczstore append data/store.vcz data/sample-part2.vcf.vcz
% uv run vcztools query -l data/store.vcz
NA00001
NA00002
NA00003

# Remove a sample from the store
% uv run vczstore remove data/store.vcz NA00002
% uv run vcztools query -l data/store.vcz
NA00001
NA00003
```

* Transactions: icechunk
* Distributed: single process
* Zarr: format 3

```shell
% uv sync --extra icechunk

# Create some VCZ data
% rm -rf data
% mkdir data
% uv run vcf2zarr convert --no-progress --zarr-format=3 --samples-chunk-size=4 tests/data/vcf/sample-part1.vcf.gz data/sample-part1.vcf.vcz
% uv run vcf2zarr convert --no-progress --zarr-format=3 tests/data/vcf/sample-part2.vcf.gz data/sample-part2.vcf.vcz

# Copy first vcz to an icechunk store
% uv run vczstore copy-store-to-icechunk data/sample-part1.vcf.vcz data/store.vcz
% rm -rf data/sample-part1.vcf.vcz

# Show the samples in each
% uv run vcztools query -l data/store.vcz --backend-storage icechunk
NA00001
NA00002
% uv run vcztools query -l data/sample-part2.vcf.vcz
NA00003

# Append data to the store
% uv run vczstore append data/store.vcz data/sample-part2.vcf.vcz --backend-storage icechunk
% uv run vcztools query -l data/store.vcz --backend-storage icechunk
NA00001
NA00002
NA00003

# Remove a sample from the store
% uv run vczstore remove data/store.vcz NA00002 --backend-storage icechunk
% uv run vcztools query -l data/store.vcz --backend-storage icechunk
NA00001
NA00003
```
