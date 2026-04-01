#!/usr/bin/env bash

# Drop samples from a VCF file and write to an output VCF
# while retaining all the fields in the header.

set -ex

INPUT_VCF=$1
OUTPUT_VCF=$2

INPUT_VCF_NAME=$(basename $INPUT_VCF)
TMP_VCF=/tmp/${INPUT_VCF_NAME}_tmp.vcf.gz
HEADER_FILE=/tmp/${INPUT_VCF_NAME}_header.txt

# save the header to preserve FORMAT fields definitions
bcftools view -h $INPUT_VCF > $HEADER_FILE

# remove the last line of the header which contains the sample names
sed -i '' -e '$d' $HEADER_FILE

# replace it with the fixed fields and INFO
echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO" >> $HEADER_FILE

# remove samples
bcftools view --drop-genotypes $INPUT_VCF -W=csi -o $TMP_VCF

# change the header to include FORMAT fields
bcftools reheader -h $HEADER_FILE $TMP_VCF | bcftools view -W=csi -o $OUTPUT_VCF
