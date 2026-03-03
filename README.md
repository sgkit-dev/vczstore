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
% conda activate vczstore-poc-zarr-v3

# Create some VCZ data
% rm -rf data
% mkdir data
% vcf2zarr convert --no-progress tests/data/vcf/sample-part1.vcf.gz data/store.vcz
% vcf2zarr convert --no-progress tests/data/vcf/sample-part2.vcf.gz data/sample-part2.vcf.vcz

# Show the samples in each
% vcztools query -l data/store.vcz
NA00001
NA00002
% vcztools query -l data/sample-part2.vcf.vcz
NA00003

# Append data to the store
% vczstore append data/store.vcz data/sample-part2.vcf.vcz
% vcztools query -l data/store.vcz
NA00001
NA00002
NA00003

# Remove a sample from the store
% vczstore remove data/store.vcz NA00002
% vcztools query -l data/store.vcz
NA00001
NA00003
```

* Transactions: icechunk
* Distributed: single process
* Zarr: v3, format 3

```shell
% conda activate vczstore-poc-icechunk

# Create some VCZ data
% rm -rf data
% mkdir data
% BIO2ZARR_ZARR_FORMAT=3 vcf2zarr convert --no-progress tests/data/vcf/sample-part1.vcf.gz data/sample-part1.vcf.vcz
% BIO2ZARR_ZARR_FORMAT=3 vcf2zarr convert --no-progress tests/data/vcf/sample-part2.vcf.gz data/sample-part2.vcf.vcz

# Copy first vcz to an icechunk store
% vczstore copy-store-to-icechunk data/sample-part1.vcf.vcz data/store.vcz
% rm -rf data/sample-part1.vcf.vcz

# Show the samples in each
% vcztools query -l data/store.vcz --zarr-backend-storage icechunk
NA00001
NA00002
% vcztools query -l data/sample-part2.vcf.vcz
NA00003

# Append data to the store
% vczstore append data/store.vcz data/sample-part2.vcf.vcz --zarr-backend-storage icechunk
% vcztools query -l data/store.vcz --zarr-backend-storage icechunk
NA00001
NA00002
NA00003

# Remove a sample from the store
% vczstore remove data/store.vcz NA00002 --zarr-backend-storage icechunk
% vcztools query -l data/store.vcz --zarr-backend-storage icechunk
NA00001
NA00003
```

* Transactions: none
* Distributed: multiple processes
* Zarr: v3, format 2

```shell
% conda activate vczstore-poc-zarr-v3

# Create some VCZ data
% rm -rf data
% mkdir data
% vcf2zarr convert --no-progress --variants-chunk-size=10 tests/data/vcf/chr22.vcf.gz data/store.vcz

# Show the samples in the store
% vcztools query -l data/store.vcz | wc -l
100

# Remove a sample from the store using 3 processes
% vczstore dremove-init data/store.vcz HG00100 -n 3
% parallel -j 3 vczstore dremove-partition data/store.vcz {} ::: seq 0 2
% vczstore dremove-finalise data/store.vcz
% vcztools query -l data/store.vcz | wc -l
99
```

### Matrix testing

All (quick)

```shell
conda activate vczstore-poc-zarr-v3
pytest -vs

conda activate vczstore-poc-zarr-v3-f3
BIO2ZARR_ZARR_FORMAT=3 pytest -vs

conda activate vczstore-poc-icechunk
BIO2ZARR_ZARR_FORMAT=3 pytest -vs -k icechunk
```

* Transactions: none
* Distributed: single process
* Zarr: v3

```shell
conda deactivate
conda env remove -n vczstore-poc-zarr-v3
conda create --name vczstore-poc-zarr-v3 -y 'python==3.12'
conda activate vczstore-poc-zarr-v3
pip install -e '.[dev]'
pytest -vs
```

* Transactions: none
* Distributed: single process
* Zarr: v3, format 3

```shell
conda deactivate
conda env remove -n vczstore-poc-zarr-v3-f3
conda create --name vczstore-poc-zarr-v3-f3 -y 'python==3.12'
conda activate vczstore-poc-zarr-v3-f3
pip install -e '.[dev]'
BIO2ZARR_ZARR_FORMAT=3 pytest -vs
```

* Transactions: icechunk
* Distributed: single process
* Zarr: v3, format 3

```shell
conda deactivate
conda env remove -n vczstore-poc-icechunk
conda create --name vczstore-poc-icechunk -y 'python==3.12'
conda activate vczstore-poc-icechunk
pip install -e '.[dev]'
pip install icechunk hypothesis
BIO2ZARR_ZARR_FORMAT=3 pytest -vs -k icechunk
```
