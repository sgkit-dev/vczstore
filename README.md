# vczstore

Initial experiments to try different technologies for a vczstore POC.

### Operations

The key operations we want to support are:

* Append to a Zarr store (atomically)
* Remove values from a Zarr store (atomically)
* Tag a Zarr store

### Support matrix

Using zarr-python directly (single machine)

| zarr-python (format) | v3 (v2)            | v3 (v3)            |
|----------------------|--------------------|--------------------|
| **no transactions**  | :white_check_mark: | :white_check_mark: |
| **icechunk**         | N/A                | :white_check_mark: |

In addition the following things are missing or not yet supported:

* Error checking (e.g. that the store and the vcz being appended have compatible fields)
* Allele harmonisation
* Cloud stores

### Demo

* Transactions: none
* Distributed: single process
* Zarr: v3, format 2

```shell
% uv sync --group dev

# Create some VCZ data
% rm -rf data
% mkdir data
% uv run vcf2zarr convert --no-progress tests/data/vcf/sample-part1.vcf.gz data/store.vcz
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
* Zarr: v3, format 3

```shell
% uv sync --extra icechunk

# Create some VCZ data
% rm -rf data
% mkdir data
% BIO2ZARR_ZARR_FORMAT=3 uv run vcf2zarr convert --no-progress tests/data/vcf/sample-part1.vcf.gz data/sample-part1.vcf.vcz
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
* Zarr: v3, format 2

```shell
% uv sync --group dev

# Create some VCZ data
% rm -rf data
% mkdir data
% uv run vcf2zarr convert --no-progress --variants-chunk-size=10 tests/data/vcf/chr22.vcf.gz data/store.vcz

# Show the samples in the store
% uv run vcztools query -l data/store.vcz | wc -l
100

# Remove a sample from the store using 3 processes
% uv run vczstore dremove-init data/store.vcz HG00100 -n 3
% parallel -j 3 uv run vczstore dremove-partition data/store.vcz {} ::: $(seq 0 2)
% uv run vczstore dremove-finalise data/store.vcz
% uv run vcztools query -l data/store.vcz | wc -l
99
```
