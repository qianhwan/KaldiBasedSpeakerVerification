// extractFeatures.cpp
// Extract MFCCs given a list of speech
// QIANHUI WAN, University of Ottawa
// Jan-26-2018
//
// Usage: extractFeatures scp,p:wav.scp ark,scp:mfcc.ark,mfcc.scp
// Given a text file wav.scp, which contains the wavefile name and wavefile path,
// this code outputs two files:
// 1. mfcc.ark -> binaries of MFCCs (including deltas and double deltas, cmvn applied)
// 2. mfcc.scp -> text of information of mfcc.ark

#include "feat/wave-reader.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "hmm/posterior.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "hmm/transition-model.h"
#include "ivector/plda.h"
#include <sstream>
#include <stdio.h>
extern "C"
{
  #include <fvad.h>
}
using namespace kaldi;

namespace SpeakerIdentification {

	bool process_vad(Vector<BaseFloat> wave_piece_buffer, Fvad *vad, size_t framelen)
 	{
    		
    		int16_t *buf1 = NULL;
    		double *buf0 = NULL;
    		int vadres = -1;
    		if (!(buf1 = (int16_t*)malloc(framelen * sizeof *buf1)) || !(buf0 = (double*)malloc(framelen * sizeof *buf0))) {
        	std::cerr << "failed to allocate buffers\n";
    		}
    		if (framelen != wave_piece_buffer.Dim())
        		std::cerr << "Wrong input\n";	

    		// Convert the read samples to int16
    		for (int i = 0; i < framelen; i++){
			buf0[i] = wave_piece_buffer(i);
    			buf1[i] = buf0[i];
    		}
    		vadres = fvad_process(vad, buf1, framelen);
    		if (vadres < 0) {
        		std::cerr << "VAD processing failed\n";
    		}
    		vadres = !!vadres; // make sure it is 0 or 1
    		if (vadres == 1){
			return true;
    		}else{
			return false;
    		}
 	}

	// Initialize the parameters for MFCC extracting
	void MfccInitiation(WaveData *wave, MfccOptions *op) {
		op->num_ceps = 20; // Number of coefficients (include energy)
		op->frame_opts.window_type = "hamming"; // window type
		op->frame_opts.frame_shift_ms = 10; // 10 ms frame shift, default
   		op->frame_opts.frame_length_ms = 25.0; // 25 ms frame length, default
  		op->frame_opts.preemph_coeff = 0.97; // pre-emphasize coefficient, 0.97 is default
		op->frame_opts.remove_dc_offset = false;
		op->frame_opts.round_to_power_of_two = true;
		op->frame_opts.samp_freq = wave->SampFreq(); // sampling rate
		op->htk_compat = false;
		op->use_energy = true;  // replace C0 with energy.
		if (RandInt(0, 1) == 0)
			op->frame_opts.snip_edges = false;
	}

	// Append Deltas and double deltas to MFCCs, then apply CMVN
	void MfccProcess(Matrix<BaseFloat> *feats, Matrix<BaseFloat> *processed_feats) {
		DeltaFeaturesOptions delta_opts; // Initialize Delta options
                delta_opts.window = 3; // delta window length 
		SlidingWindowCmnOptions cmvn_opts; // Initialize CMVN options
                cmvn_opts.cmn_window = 300; // CMVN window length
                cmvn_opts.normalize_variance = true; // apply variance normalization
                cmvn_opts.center = true; // use a window centered on the current frame
		Matrix<BaseFloat> delta_feats;
		ComputeDeltas(delta_opts, *feats, &delta_feats); // Append deltas and double deltas
		Matrix<BaseFloat> cmvn_feats_(delta_feats.NumRows(),
			delta_feats.NumCols(), kUndefined);
		SlidingWindowCmn(cmvn_opts, delta_feats, &cmvn_feats_); // Apply sliding window CMVN
		*processed_feats = cmvn_feats_;
	}

}  // end namespace SpeakerIdentification

using namespace SpeakerIdentification;

int main(int argc, char *argv[]) {
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;
	// Read input
	if (argc != 3){
		std::cerr << "Usage: extractFeatures scp,p:data/wav.scp ark,scp:feat/mfcc.ark,feat/mfcc.scp!\n";
	}
	std::string wav_scp = argv[1]; // wav.scp
	std::string output_filename = argv[2];
	SequentialTableReader<WaveHolder> reader(wav_scp);
	BaseFloatMatrixWriter feat_writer(output_filename);
	int num_utts = 0;
	for (; !reader.Done(); reader.Next()) {
		num_utts++;
		std::string uttid = reader.Key();
		const WaveData &wave_data = reader.Value();
		WaveData wave = wave_data; 		
		KALDI_ASSERT(wave.Data().NumRows() == 1);
		SubVector<BaseFloat> waveform(wave.Data(), 0);
	 	Vector<BaseFloat> waveform_vad;
		//waveform_vad.CopyFromVec(waveform);
		Vector<BaseFloat> wave_piece_buffer;

		// vad
		int32 wave_length = waveform.Dim(), // audio length
 		audio_segment = 0.01 * wave.SampFreq(), // # of samples for 10ms
 		num_segment = wave_length / audio_segment, // # of 10ms segments in total
 		offset_start = 0;
 		size_t framelen = audio_segment;
 		Fvad *vad = NULL;
 		vad = fvad_new();
 		int mode = 3;
 		if (fvad_set_sample_rate(vad, wave.SampFreq()) < 0) {
  			std::cerr <<"invalid sample rate: "<< wave.SampFreq() << " Hz\n";
 		}
 		if (fvad_set_mode(vad, mode) < 0) {
  			std::cerr <<"invalid mode: "<< mode << "\n";
 		}
 		int32 vad_seg = 0;
 		// vad every 10ms
 		for (int32 i = 0; i < num_segment; i++) {
  			wave_piece_buffer = waveform.Range(offset_start, audio_segment);// fill wave_piece_buffer with new 10ms 
  			if (process_vad(wave_piece_buffer, vad, framelen)){ // do vad, discard if vad returns false
				vad_seg++;
				waveform_vad.Resize(audio_segment * vad_seg, kCopyData);
				for (int32 idx = 0; idx < audio_segment; idx++)
					waveform_vad((vad_seg - 1) * audio_segment + idx) = wave_piece_buffer(idx);
  			}
			offset_start += audio_segment;
 		}
		// the parametrization object
		MfccOptions op_mfcc;
		MfccInitiation(&wave, &op_mfcc);
		Mfcc mfcc(op_mfcc);
		Matrix<BaseFloat> features;
		mfcc.Compute(waveform_vad, 1.0, &features);
		Matrix<BaseFloat> processed_feats;
		MfccProcess(&features, &processed_feats);
		feat_writer.Write(uttid, processed_feats);
		std::cout << "Done extracting features from " << uttid << "\n";
	}
	std::cout << "Done with " << num_utts << " speech!\n";
}
