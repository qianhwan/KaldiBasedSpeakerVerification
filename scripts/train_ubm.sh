#!/bin/bash
# 
# This script will train a UBM with the following parameters:
# number of mixtures = 1024
# minimum Gaussian weight = 0.0001
# Iteration times = 4
# This will output two files in ../mat: final.dubm, final.ubm
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
  echo "Usage: local/train_ubm.sh <feat-dir> <ubm-dir>"
  echo "Trains a full-covariance UBM "
  exit 1;
fi

featdir=$1
ubmdir=$2
tempdir=$ubmdir/temp

kaldigmmdir=/home/qianhuiwan/sourcecodes/kaldi/src/gmmbin
kaldifgmmdir=/home/qianhuiwan/sourcecodes/kaldi/src/fgmmbin
mkdir -p $tempdir
mkdir -p $ubmdir

# parameters
min_gauss_w=0.00001
nb_mix=1024
nb_iter=5
num_gselect=20 # cutoff for Gaussian-selection that we do once at the start.

$kaldigmmdir/gmm-global-init-from-feats --num-frames=500000 --min-gaussian-weight=$min_gauss_w \
--num-gauss=$nb_mix --num-gauss-init=$num_gselect --num-iters=$nb_iter \
ark:$featdir/mfcc.ark $tempdir/0.dubm

$kaldigmmdir/gmm-gselect --n=$num_gselect $tempdir/0.dubm ark:$featdir/mfcc.ark ark:$tempdir/g-select.ark

for x in `seq 0 $[$nb_iter-1]`; do
	echo "$0: Training pass $x"
	$kaldigmmdir/gmm-global-acc-stats --gselect=ark,s,cs:$tempdir/g-select.ark \
	$tempdir/$x.dubm ark:$featdir/mfcc.ark $tempdir/$x.acc || exit 1;

	if [ $x -lt $[$num_iters-1] ]; then # Don't remove low-count Gaussians till last iter,
      	opt="--remove-low-count-gaussians=false" # or gselect info won't be valid any more.
   	else
      	opt="--remove-low-count-gaussians=true"
   	fi

	$kaldigmmdir/gmm-global-est $opt --min-gaussian-weight=$min_gauss_w $tempdir/$x.dubm $tempdir/$x.acc \
     	$tempdir/$[$x+1].dubm || exit 1;
done
mv $tempdir/$nb_iter.dubm $ubmdir/final.dubm || exit 1;
rm $tempdir/*

$kaldigmmdir/gmm-global-to-fgmm $ubmdir/final.dubm $tempdir/0.ubm

$kaldifgmmdir/fgmm-global-to-gmm $tempdir/0.ubm $tempdir/0.dubm

$kaldigmmdir/gmm-gselect --n=$num_gselect $tempdir/0.dubm ark:$featdir/mfcc.ark ark:$tempdir/g-select.ark

for x in `seq 0 $[$nb_iter-1]`; do
	echo "$0: Training pass $x"
	$kaldifgmmdir/fgmm-global-acc-stats --gselect=ark,s,cs:$tempdir/g-select.ark \
	$tempdir/$x.ubm ark:$featdir/mfcc.ark $tempdir/$x.acc || exit 1;

	if [ $x -lt $[$num_iters-1] ]; then # Don't remove low-count Gaussians till last iter,
      	opt="--remove-low-count-gaussians=false" # or gselect info won't be valid any more.
   	else
      	opt="--remove-low-count-gaussians=true"
   	fi

	$kaldifgmmdir/fgmm-global-est $opt --min-gaussian-weight=$min_gauss_w $tempdir/$x.ubm $tempdir/$x.acc \
     	$tempdir/$[$x+1].ubm || exit 1;
done
mv $tempdir/$nb_iter.ubm $ubmdir/final.ubm || exit 1;
rm $tempdir/*

