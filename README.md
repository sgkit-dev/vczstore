# vczstore

Tools for managing a VCF Zarr store.

### Introduction

The [VCF Zarr format](https://github.com/sgkit-dev/vcf-zarr-spec/) enables efficient, scalable storage and analysis of large-scale genetic variation data.

VCF Zarr files can be created using [bio2zarr](https://sgkit-dev.github.io/bio2zarr/intro.html). While ideal for one-off conversion of VCF files to Zarr, bio2zarr does not support updating a VCF Zarr store with new samples.

Vczstore solves the update use case by providing the following operations
1. **Append** new samples to a Zarr store
2. **Remove** samples from a Zarr store

The append operation works by appending a VCF Zarr file to the store (which is another VCF Zarr). The file being appended is created using bio2zarr.

### Variant sets

Note that only new _samples_ can be added - not new variants - so the store contains a fixed set of variants that must be known before it is created. This is usually not a limitation since the samples come from the same genotype array, or even from the same reference panel for imputed data.

If the source VCFs have different (but typically overlapping) sets of variants, then they need to be harmonised with the full set of variants before being converted to VCF Zarr. This can be accomplished using bcftools merge or similar tools - it is not performed by vczstore itself.

The append operation will perform a check that the contig, position and allele (REF and ALT) fields all match before performing the update. It will fail if there is a mismatch, so samples with inconsistent variant sets cannot be appended. The check is strict - allele ordering must match exactly too.

Multiallelic sites, and split alleles (mutiple records for a site) are both accepted, as long as the ordering is consistent for all source VCFs. 

### Implementation

The implementation uses zarr-python (version 3) directly to update Zarr chunks in the store. Stores using Zarr format 2 and 3 are supported, as well as Icechunk storage. Using Icechunk means that updates are performed in a transaction so that other users accessing the store will see consistent updates.

| zarr-python (format) | v2                 | v3                 |
|----------------------|--------------------|--------------------|
| **no transactions**  | :white_check_mark: | :white_check_mark: |
| **icechunk**         | N/A                | :white_check_mark: |

VCF Zarr stores can reside in cloud stores such as Amazon S3 or Azure Cloud Storage.

There is a multiple process implementation (like bio2zarr's [distributed implementation](https://sgkit-dev.github.io/bio2zarr/vcf2zarr/tutorial.html#large-dataset)) for running on large datasets. See below for an example.

### Demo

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
% BIO2ZARR_ZARR_FORMAT=3 uv run vcf2zarr convert --no-progress --samples-chunk-size=4 tests/data/vcf/sample-part1.vcf.gz data/sample-part1.vcf.vcz
% BIO2ZARR_ZARR_FORMAT=3 uv run vcf2zarr convert --no-progress tests/data/vcf/sample-part2.vcf.gz data/sample-part2.vcf.vcz

# Copy first vcz to an icechunk store
% uv run vczstore copy-store-to-icechunk data/sample-part1.vcf.vcz data/store.vcz
% rm -rf data/sample-part1.vcf.vcz

# Show the samples in each
% uv run vcztools query -l data/store.vcz --zarr-backend-storage icechunk
NA00001
NA00002
% uv run vcztools query -l data/sample-part2.vcf.vcz
NA00003

# Append data to the store
% uv run vczstore append data/store.vcz data/sample-part2.vcf.vcz --zarr-backend-storage icechunk
% uv run vcztools query -l data/store.vcz --zarr-backend-storage icechunk
NA00001
NA00002
NA00003

# Remove a sample from the store
% uv run vczstore remove data/store.vcz NA00002 --zarr-backend-storage icechunk
% uv run vcztools query -l data/store.vcz --zarr-backend-storage icechunk
NA00001
NA00003
```

* Transactions: none
* Distributed: multiple processes
* Zarr: format 2

```shell
% uv sync --group dev

# Create some VCZ data
% rm -rf data
% mkdir data
% uv run vcf2zarr convert --no-progress --variants-chunk-size=10 --samples-chunk-size=50 tests/data/vcf/chr22-part1.vcf.gz data/store.vcz
% uv run vcf2zarr convert --no-progress --variants-chunk-size=10 --samples-chunk-size=50 tests/data/vcf/chr22-part2.vcf.gz data/chr22-part2.vcf.vcz

# Show the number of samples in each
% uv run vcztools query -l data/store.vcz | wc -l
55
% uv run vcztools query -l data/chr22-part2.vcf.vcz | wc -l
45

# Append data to the store using 3 processes
% uv run vczstore dappend-init data/store.vcz data/chr22-part2.vcf.vcz -n 3
% parallel -j 3 uv run vczstore dappend-partition data/store.vcz data/chr22-part2.vcf.vcz {} ::: $(seq 0 2)
% uv run vczstore dappend-finalise data/store.vcz data/chr22-part2.vcf.vcz
% uv run vcztools query -l data/store.vcz | wc -l
100

# Remove a sample from the store using 3 processes
% uv run vczstore dremove-init data/store.vcz HG00100 -n 3
% parallel -j 3 uv run vczstore dremove-partition data/store.vcz {} ::: $(seq 0 2)
% uv run vczstore dremove-finalise data/store.vcz
% uv run vcztools query -l data/store.vcz | wc -l
99
```
