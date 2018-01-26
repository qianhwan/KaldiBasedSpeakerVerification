#!/bin/bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# Modified 2017  Qianhui Wan
# This script will create three text files with speaker/speech information, given 
# the input path and output path.
# The input path should be a folder contains speaker speech of the following structure:
# input_path/speaker_id/speech_01.wav
#		       /speech_02.wav
#                      ......
# The three outputs are:
# 1. wav.scp, contains the file name and path of speech,
# 2. utt2spk, gives the speaker id of each speech
# 3. spk2utt, gives all speech file names from one speaker id.


if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <data-dir> <output-dir>"
  echo "e.g.: $0 /media/sf_camera_iwatchlife2017/dev_mix dev_data"
  exit 1
fi

src=$1
dst=$2

mkdir -p $dst || exit 1;

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1;
[ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1;


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk

for reader_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sort); do
  reader=$(basename $reader_dir)
  find -L $reader_dir/ -iname "*.wav" | sort | xargs -I% basename % .wav | \
      awk -v "dir=$reader_dir" '{printf "%s %s/%s.wav\n", $0, dir, $0}' >>$wav_scp|| exit 1
  find -L $reader_dir/ -iname "*.wav" | sort | xargs -I% basename % .wav | \
      awk -v "reader=$reader" '{printf "%s %s\n", $0, reader}'  >>$utt2spk || exit 1
done

spk2utt=$dst/spk2utt
./utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

#validate_data_dir.sh --no-feats $dst || exit 1;

echo "$0: successfully prepared data in $dst"

exit 0
