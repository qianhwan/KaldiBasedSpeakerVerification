#!/bin/bash
# KaldiBasedSpeakerVerification
# test1Test.sh
# ========================================
# Author: Qianhui Wan
# Version: 1.0.0
# Date   : 2018-01-23
# ========================================
# The following lines will setup the path to each lib
# path to kaldi/src/lib
export LD_LIBRARY_PATH=/home/qianhuiwan/sourcecodes/kaldi/src/lib:$LD_LIBRARY_PATH
# path to altas
export LD_LIBRARY_PATH=/usr/lib64/atlas:$LD_LIBRARY_PATH
# path to openfst
export LD_LIBRARY_PATH=/home/qianhuiwan/sourcecodes/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH
# path to usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


../src/identifySpeaker ./example_data/test/84/84-121550-0030.wav

