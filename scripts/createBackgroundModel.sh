#!/bin/bash
# KaldiBasedSpeakerVerification
# createBackGroundModel.sh
# ========================================
# Author: Qianhui Wan
# Version: 1.0.0
# Date   : 2018-01-23
# This process is modified from Kaldi examples: sre08 and sre10
# ========================================
# This script will train the background models for a speaker verification system
# final.ubm, final.dubm -> Universal background model
# final.ie -> i-vector extractor
# final.plda -> Probablistic Linear Discriminant Analysis model
# transform.mat -> Linear Discriminant Analysis, Within Class Covariance Normalization model
# mean.vec -> global mean of background i-vectors
# All models will be stored in ../mat

# The following lines will setup the path to each lib
# path to kaldi/src/lib
export LD_LIBRARY_PATH=/home/qianhuiwan/sourcecodes/kaldi/src/lib:$LD_LIBRARY_PATH
# path to altas
export LD_LIBRARY_PATH=/usr/lib64/atlas:$LD_LIBRARY_PATH
# path to openfst
export LD_LIBRARY_PATH=/home/qianhuiwan/sourcecodes/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH
# path to usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# path to directory that contains development dataset, see details in README for the right format of storing dataset.
speechdir=/media/sf_LibriSpeech/thesis/dev_clean_small
datadir=../dev_data
matdir=../mat
funcdir=../src
mkdir -p $datadir
mkdir -p $matdir
# Step 1. Prepare speech data information (wav.scp, utt2spk, spk2utt)
#./data_prep.sh $speechdir $datadir

# Step 2. Extract mfcc (plus delta and double delta, apply cmvn)
#$funcdir/extractFeatures scp,p:$datadir/wav.scp ark,scp:$datadir/mfcc.ark,$datadir/mfcc.scp;

# Step 3. Train UBM
#./train_ubm.sh $datadir $matdir

# Step 4. Train i-vector extractor
#./train_ivextractor.sh $datadir $matdir

# Step 5. Train LDA, WCCN and PLDA
./train_comp_plda.sh $datadir $matdir

echo "Done Training Background Models!"
