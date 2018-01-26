#!/bin/bash
# 
# This script will train a transform matrix of LDA and WCCN as well as a PLDA model with the following parameters:
# dimension of PLDA = 150
# This will output three files in ../mat: final.plda, transform.mat, mean_vec

codedir=`pwd`
parentdir="$(dirname "$codedir")"
# The following lines will setup the path to each lib
# path to kaldi/src/lib
export LD_LIBRARY_PATH=/home/qianhuiwan/sourcecodes/kaldi/src/lib:$LD_LIBRARY_PATH
# path to altas
export LD_LIBRARY_PATH=/usr/lib64/atlas:$LD_LIBRARY_PATH
# path to openfst
export LD_LIBRARY_PATH=/home/qianhuiwan/sourcecodes/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH
# path to usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

if [ $# != 2 ]; then
  echo "Usage: local/train_comp_plda.sh <feat-dir> <output-dir>"
  echo "Trains a transform matrix of LDA and WCCN as well as a PLDA model"
  exit 1;
fi

featdir=$1
pldadir=$2
tempdir=$pldadir/temp

kaldigmmdir=/home/qianhuiwan/sourcecodes/kaldi/src/gmmbin
kaldifgmmdir=/home/qianhuiwan/sourcecodes/kaldi/src/fgmmbin
kaldiivectordir=/home/qianhuiwan/sourcecodes/kaldi/src/ivectorbin
kaldiscaledir=/home/qianhuiwan/sourcecodes/kaldi/src/bin
mkdir -p $tempdir

# parameters
plda_dim=150
num_gselect=20 # cutoff for Gaussian-selection that we do once at the start.
min_post=0.025

# Extract ivs
$kaldifgmmdir/fgmm-global-to-gmm $pldadir/final.ubm $tempdir/final.dubm

$kaldigmmdir/gmm-gselect --n=$num_gselect $tempdir/final.dubm ark:$featdir/mfcc.ark ark:$tempdir/g-select.ark

$kaldifgmmdir/fgmm-global-gselect-to-post --min-post=$min_post $pldadir/final.ubm \
ark:$featdir/mfcc.ark ark,s,cs:$tempdir/g-select.ark ark:$tempdir/post0.ark

$kaldiscaledir/scale-post ark:$tempdir/post0.ark 1.0 ark:$tempdir/post.ark

$kaldiivectordir/ivector-extract --verbose=0 $pldadir/final.ie ark:$featdir/mfcc.ark ark:$tempdir/post0.ark \
ark,scp,t:$featdir/ivector.ark,$featdir/ivector.scp

# IV length normalization
$kaldiivectordir/ivector-normalize-length scp:$featdir/ivector.scp ark,scp:$featdir/ivector_ln1.ark,$featdir/ivector_ln1.scp

# Train LDA + WCCN
$kaldiivectordir/ivector-compute-lda --dim=$plda_dim --total-covariance-factor=0.1 ark:$featdir/ivector_ln1.ark \
ark:$featdir/utt2spk $pldadir/transform.mat

$kaldiivectordir/ivector-transform $pldadir/transform.mat ark:$featdir/ivector_ln1.ark ark:$featdir/ivector_lw.ark

# Train PLDA
$kaldiivectordir/ivector-compute-plda ark:$featdir/spk2utt ark:$featdir/ivector_lw.ark $tempdir/plda_lw

$kaldiivectordir/ivector-copy-plda --smoothing=0.0 $tempdir/plda_lw $pldadir/final.plda

$kaldiivectordir/ivector-mean ark:$featdir/ivector_lw.ark $pldadir/mean_lw.vec

rm -r $tempdir


