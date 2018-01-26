KaldiBasedSpeakerVerification
========================================
Author: Qianhui Wan
Version: 1.0.0
Date   : 2018-01-23

Prerequisite
------------
1. Kaldi 5.3, as well as Altas and OpenFst required by Kaldi.
https://github.com/kaldi-asr/kaldi

2. libfvad, Voice activity detection (VAD) library, based on WebRTC's VAD engine.
https://github.com/dpirch/libfvad

Installation
------------
1. Install Kaldi 5.3:
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi

2. Install Kaldi's required libraries:
cd to /kaldi/tools and follow INSTALL instructions there.

3. Compile and finish Kaldi install:
cd to /kaldi/src and follow INSTALL instructions there.

4. Install libfvad:
git clone https://github.com/dpirch/libfvad
cd libfvad
./bootstrap
./configure
make
make install (perhaps sudo at this command)

5. Install KaldiBasedSpeakerVerification

cd KaldiBasedSpeakerVerification/src
*edit makefile; provide the correct locations for this project and the libraries.
make 
(This will output 3 executables under /src: enroll, identifySpeaker and extractFeatures)


Project file structure (under KaldiBasedSpeakerVerification folder)
----------------------------------
/examples
 contains enroll and test examples, along with example data

/examples/iv
 contains i-vector features extracted from enrollment.(this can be empty before enrolling speakers, must have 2 files before testing)
 
/examples/mat
 contains background model data, must have six files.
 
/scripts
 contains scripts mainly used to create background model.
 
/src
 contains code for 3 applications: creating a background model, enrolling speakers and speaker identification.



Main applications
-------------------------------------------------
/src/enroll.cpp
 This program is used to extract speech features from one speaker.
 Usage: enroll speakerId wavefile
 The output should look like:
 Not registered speaker: speakerId. Created a new spkid
 or
 Found registered speaker: speakerId. Updated speaker model

 The wavefile should be in .wav format.

 This will create/update two files in /iv: train_iv.ark and train_num_utts.ark.

/src/identifySpeaker.cpp
 This program process a given audio clip and output person identification every ~3.2 seconds.
 Usage: identifySpeaker wavefile
 The output should look like:
 Family membmer detected! Speaker: 225
 Family membmer detected! Speaker: 225
 Stanger detected!
 Family membmer detected! Speaker: 227
 Family membmer detected! Speaker: 227
 ...

 It will also output the probability score for each segments -> this could be used to adjust the decision threshold due to different audio condition.


Examples
-------------------------------------------------
After installing all required applications, you can run the following examples to test if your installation is right.

1. make sure there is three folder in /examples
  /example_data
  /iv
  /mat (due to the file size limit of GitHub, final.ie was zipped into several parts. To unzip, do: cat iepart* -> final.ie)

2. run ./test1Enroll.sh
This will enroll all speech files in /example_data/enroll
The output should look like:

The total active speech is 1.61 seconds.
No registered speaker: 174. Create a new spkid
Done.
The total active speech is 15 seconds.
Found registered speaker: 174. Update speaker model
Done.
The total active speech is 0.88 seconds.
No registered speaker: 84. Create a new spkid
Done.
The total active speech is 3.47 seconds.
Found registered speaker: 84. Update speaker model
Done.

3. run ./test1Test.sh
This will test speech /example_data/test/84/84-121550-0030.wav against all registered speaker
The output should look like:

Effective speech length: 2.605s.No family member detected.		(score: 4.97931)
Effective speech length: 5.685s.Family member detected! Speaker: 84	(score: 33.7779)
Speech data is finished!
Done.


*Note:
There will also be outputs of kaldi log which look like:
LOG ([5.3.96~1-7ee7]:ComputeDerivedVars():ivector-extractor.cc:183) Computing derived variables for iVector extractor
LOG ([5.3.96~1-7ee7]:ComputeDerivedVars():ivector-extractor.cc:204) Done.

This tells you one audio segment has been processed and can be omitted by setting kaldi verbose level.

Background Model Training
-------------------------------------
/src/extractFeatures
 The program extracts 20-dim MFCC (with energy), append deltas and double deltas, and apply CMVN
 Usage: extractFeatures wav.scp ark,scp:feat.ark,feat.scp
 Input: wav.scp, a text list of speech file name and path
 Output: feat.ark, feat.scp -> same as kaldi.

/scripts/data_prep.sh
 usage: data_prep.sh path_to_speech path_to_info
 prepare useful text file for later process, please refer to data_prep.sh for details

/scripts/utt2spk_to_spk2utt.pl
 usage: utt2spk_to_spk2utt.pl utt2spk > spk2utt 
 create the spk2utt file with given utt2spk file

/scripts/train_ubm.sh
 usage: train_ubm.sh path_to_feat path_to_mat
 output: final.dubm, final.ubm
 please refer to train_ubm.sh for details
 
/scripts/train_ivextractor.sh
 usage: train_ivextractor.sh path_to_feat path_to_mat
 output: final.ie
 please refer to train_ivextractor.sh for details
 
/scripts/train_comp_plda.sh
 usage: train_comp_plda.sh path_to_feat path_to_mat
 output: final.plda, transform.mat, mean_vec
 please refer to train_comp_plda.sh for details

The following folders will be created during running: 
 /dev_data
 contains development dataset speech information, MFCC features and i-vectors 
 
 /mat
 contains all trained models:
 final.dubm, final.ubm, final.ie, final.plda, transform.mat, mean_vec

Note: The whole process can take several hours (e.g. 5 to 6 hours from VirtualBox-run CentOS version).
Note: All scripts need to modified manually for the path (same as examples), this can be avoided if you add all paths to environmental variables.

