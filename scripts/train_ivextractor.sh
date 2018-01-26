#!/bin/bash
# 
# This script will train a i-vector extractor with the following parameters:
# dimension of i-vector = 400
# Iteration times = 5
# This will output one file in ../mat: final.ie
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
  echo "Usage: local/train_ivextractor.sh <feat-dir> <ie-dir>"
  echo "Trains a i-vector extractor "
  exit 1;
fi

featdir=$1
iedir=$2
tempdir=$iedir/temp

kaldigmmdir=/home/qianhuiwan/sourcecodes/kaldi/src/gmmbin
kaldifgmmdir=/home/qianhuiwan/sourcecodes/kaldi/src/fgmmbin
kaldiivectordir=/home/qianhuiwan/sourcecodes/kaldi/src/ivectorbin
kaldiscaledir=/home/qianhuiwan/sourcecodes/kaldi/src/bin
mkdir -p $tempdir

# parameters
iv_dim=400
nb_iter=5
num_gselect=20 # cutoff for Gaussian-selection that we do once at the start.
min_post=0.025

$kaldiivectordir/ivector-extractor-init --ivector-dim=$iv_dim --use-weights=false $iedir/final.ubm $tempdir/0.ie

$kaldifgmmdir/fgmm-global-to-gmm $iedir/final.ubm $tempdir/0.dubm

$kaldigmmdir/gmm-gselect --n=$num_gselect $tempdir/0.dubm ark:$featdir/mfcc.ark ark:$tempdir/g-select.ark

$kaldifgmmdir/fgmm-global-gselect-to-post --min-post=$min_post $iedir/final.ubm \
ark:$featdir/mfcc.ark ark,s,cs:$tempdir/g-select.ark ark:$tempdir/post0.ark

$kaldiscaledir/scale-post ark:$tempdir/post0.ark 1.0 ark:$tempdir/post.ark

opt="--num-samples-for-weights=3";
for x in `seq 0 $[$nb_iter-1]`; do
	echo "$0: Training pass $x"
	$kaldiivectordir/ivector-extractor-acc-stats $opt \
	$tempdir/$x.ie ark:$featdir/mfcc.ark ark,s,cs:$tempdir/post.ark $tempdir/$x.acc || exit 1;

	$kaldiivectordir/ivector-extractor-est $tempdir/$x.ie $tempdir/$x.acc \
     	$tempdir/$[$x+1].ie || exit 1;
done
mv $tempdir/$nb_iter.ie $iedir/final.ie || exit 1;
rm $tempdir/*

