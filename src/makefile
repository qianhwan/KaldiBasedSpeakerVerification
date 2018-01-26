# Make file to build KaldiBasedSpeakerVerification
# ========================================
# Author: Qianhui Wan
# Version: 1.0.0
# Date   : 2018-01-23
# ========================================

# Variables
# path to project
PROJECT_LOCATION = /home/qianhuiwan/sourcecodes/KaldiBasedSpeakerVerification
# path to kaldi
KALDI_LOCATION = /home/qianhuiwan/sourcecodes/kaldi
OPENFST_LOCATION = $(KALDI_LOCATION)/tools/openfst
ATLAS_HEADERLOCATION = $(KALDI_LOCATION)/tools/ATLAS
ATLAS_LIBLOCATION = /usr/lib64
# path to libfvad
WEBRTC_LIBLOCATION = /home/qianhuiwan/sourcecodes/libfvad

all : enroll identifySpeaker extractFeatures

enroll : enroll.cpp
	g++ -std=c++11 -I$(KALDI_LOCATION)/src -I$(OPENFST_LOCATION)/include -DHAVE_ATLAS -I$(ATLAS_HEADERLOCATION)/include -I$(PROJECT_LOCATION)/src -I$(PROJECT_LOCATION)/src/extension -I$(WEBRTC_LIBLOCATION)/include -I. -o enroll enroll.cpp -L$/usr/local/lib -L$(KALDI_LOCATION)/src/lib -L$(OPENFST_LOCATION)/lib -L$(ATLAS_LIBLOCATION)/atlas -lkaldi-base -lkaldi-feat -lkaldi-util -lkaldi-matrix -lkaldi-hmm -lkaldi-gmm -lkaldi-ivector -lpthread -lkaldi-transform -lkaldi-tree -lfst -lfvad

identifySpeaker : identifySpeaker.cpp
	g++ -std=c++11 -I$(KALDI_LOCATION)/src -I$(OPENFST_LOCATION)/include -DHAVE_ATLAS -I$(ATLAS_HEADERLOCATION)/include -I$(PROJECT_LOCATION)/src -I$(PROJECT_LOCATION)/src/extension -I$(WEBRTC_LIBLOCATION)/include -I. -o identifySpeaker identifySpeaker.cpp -L$/usr/local/lib -L$(KALDI_LOCATION)/src/lib -L$(OPENFST_LOCATION)/lib -L$(ATLAS_LIBLOCATION)/atlas -lkaldi-base -lkaldi-feat -lkaldi-util -lkaldi-matrix -lkaldi-hmm -lkaldi-gmm -lkaldi-ivector -lpthread -lkaldi-transform -lkaldi-tree -lfst -lfvad

extractFeatures : extractFeatures.cpp
	g++ -std=c++11 -I$(KALDI_LOCATION)/src -I$(OPENFST_LOCATION)/include -DHAVE_ATLAS -I$(ATLAS_HEADERLOCATION)/include -I$(PROJECT_LOCATION)/src -I$(PROJECT_LOCATION)/src/extension -I$(WEBRTC_LIBLOCATION)/include -I. -o extractFeatures extractFeatures.cpp -L$/usr/local/lib -L$(KALDI_LOCATION)/src/lib -L$(OPENFST_LOCATION)/lib -L$(ATLAS_LIBLOCATION)/atlas -lkaldi-base -lkaldi-feat -lkaldi-util -lkaldi-matrix -lkaldi-hmm -lkaldi-gmm -lkaldi-ivector -lpthread -lkaldi-transform -lkaldi-tree -lfst -lfvad
