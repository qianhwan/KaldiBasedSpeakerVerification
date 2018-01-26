// enroll.cpp
// Registering/Updating speakers
// QIANHUI WAN, University of Ottawa
// Jan-24-2018
//
// Usage: enroll speakerid speech.wav
// Given a speaker id and a speech, this code will:
// 1. Create a new speaker in system if the speaker id is new, or
// 2. Update a existed speaker name if the speaker id is already registered.

// The following code will create two files:
// 1. train_iv.ark, it contains binaries of speaker i-vectors. One speech will lead to
// one i-vector, and for a speaker who has several speech registered, the system will
// keep the summation of all i-vectors.
// 2. train_num_utts.ark, text file that contains the # of registered speech of each speaker

// Note:
// 1. There must be a folder called /iv, it can be empty if no speaker is registered.
// 2. The audio file must be in the format of .wav (TODO: Add decoder to support other audio format)

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
#include "ivector/voice-activity-detection.h"
#include <sstream>
#include <stdio.h>
extern "C"
{
	#include <fvad.h>
}

using namespace kaldi;

namespace SpeakerIdentification {
	// Pre-trained mathematics
	DiagGmm gmm; // Diagonal covariance UBM, pre-trained model
	FullGmm fgmm; // Full covariance UBM, pre-trained model
	IvectorExtractor extractor; // I-vector extractor, pre-trained model
	Matrix<BaseFloat> transform; // LDA + WCCN compensation model, pre-trained model
	Vector<BaseFloat> mean; // Global mean of i-vectors, pre-trained model
	Plda plda; // PLDA, pre-trained model

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

	// Select Precompute Gaussian indices for pruning
	// For each frame, gives a list of the n best Gaussian indices sorted from best to worst
	void SelectPost(Matrix<BaseFloat> *feats, Posterior *post) {
		using namespace kaldi;
		using std::vector;
		typedef kaldi::int32 int32;
		typedef kaldi::int64 int64;
		int32 num_gselect = 20;
		KALDI_ASSERT(num_gselect > 0);
		BaseFloat min_post = 0.025;
		int64 tot_posts = 0;
		Matrix<BaseFloat> feats_ = *feats;
		vector<vector<int32> > gselect(feats_.NumRows());
		gmm.GaussianSelection(feats_, num_gselect, &gselect);
		int32 num_frames = feats_.NumRows();
		if (static_cast<int32>(gselect.size()) != num_frames) {
			KALDI_WARN << "gselect information for has wrong size " << 
				gselect.size() << " vs. " << num_frames;
		}
		Posterior post_(num_frames);
		bool utt_ok = true;
		for (int32 t = 0; t < num_frames; t++) {
			SubVector<BaseFloat> frame(feats_, t);
			const std::vector<int32> &this_gselect = gselect[t];
			KALDI_ASSERT(!gselect[t].empty());
			Vector<BaseFloat> loglikes;
			fgmm.LogLikelihoodsPreselect(frame, this_gselect, &loglikes);
			loglikes.ApplySoftMax();
			// now "loglikes" contains posteriors.
			if (fabs(loglikes.Sum() - 1.0) > 0.01) {
				utt_ok = false;
			}
			else {
				if (min_post != 0.0) {
					int32 max_index = 0; // in case all pruned away...
					loglikes.Max(&max_index);
					for (int32 i = 0; i < loglikes.Dim(); i++)
						if (loglikes(i) < min_post)
							loglikes(i) = 0.0;
					BaseFloat sum = loglikes.Sum();
					if (sum == 0.0) {
						loglikes(max_index) = 1.0;
					}
					else {
						loglikes.Scale(1.0 / sum);
					}
				}
				for (int32 i = 0; i < loglikes.Dim(); i++) {
					if (loglikes(i) != 0.0) {
						post_[t].push_back(std::make_pair(this_gselect[i], loglikes(i)));
						tot_posts++;
					}
				}
				if (!utt_ok) {
					KALDI_WARN << "Skipping utterance because bad posterior-sum encountered (NaN?)";
				}
				KALDI_ASSERT(!post_[t].empty());
			}
		}
		*post = post_;
	}
	
	// Extract i-vectors 
	void ExtractIvectors(Matrix<BaseFloat> *feats, Posterior *post, Vector<BaseFloat> *ivector) {
		using namespace kaldi;
		using std::vector;
		typedef kaldi::int32 int32;
		typedef kaldi::int64 int64;
		IvectorEstimationOptions opts;
		Vector<BaseFloat> ivector_;
		Vector<double> iv;
		// Parameters
		double auxf_change;
		double tot_auxf_change = 0.0;
		// Initialize
		const Matrix<BaseFloat> &mat = *feats;
		// Compute i-vectors
		if (static_cast<int32>(post->size()) != mat.NumRows()) {
			KALDI_WARN << "Size mismatch between posterior and features ";
		}
		double this_t = opts.acoustic_weight * TotalPosterior(*post),
			max_count_scale = 1.0;
		if (opts.max_count > 0 && this_t > opts.max_count) {
			max_count_scale = opts.max_count / this_t;
			KALDI_LOG << "Scaling stats by scale "
				<< max_count_scale << " due to --max-count="
				<< opts.max_count;
			this_t = opts.max_count;
		}
		ScalePosterior(opts.acoustic_weight * max_count_scale, post);
		// note: now, this_t == sum of posteriors.
		// modified
		bool need_2nd_order_stats = false;
		IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
			extractor.FeatDim(),
			need_2nd_order_stats);
		utt_stats.AccStats(mat, *post);
		iv.Resize(extractor.IvectorDim());
		iv(0) = extractor.PriorOffset();
		double old_auxf = extractor.GetAuxf(utt_stats, iv);
		extractor.GetIvectorDistribution(utt_stats, &iv, NULL);
		double new_auxf = extractor.GetAuxf(utt_stats, iv);
		auxf_change = new_auxf - old_auxf;
		tot_auxf_change += auxf_change;
		iv(0) -= extractor.PriorOffset();
		
		*ivector = Vector<BaseFloat>(iv);
	}

	// Update/Create speaker model for speaker id
	void UpdateSpkModel(std::string spkid, Vector<BaseFloat> *ivector) {
		using namespace kaldi;
		using std::vector;
		typedef kaldi::int32 int32;
		typedef kaldi::int64 int64;
		std::string old_ivector_specifier = "ark:iv//train_iv.ark",
			old_num_utts_specifier = "ark:iv//train_num_utts.ark",
			new_ivector_specifier = "ark,scp:iv//train_iv_new.ark,iv//train_iv_new.scp",
			new_num_utts_specifier = "ark,t:iv//train_num_utts_new.ark";
		SequentialBaseFloatVectorReader train_ivector_reader(old_ivector_specifier);
		RandomAccessInt32Reader num_utts_reader(old_num_utts_specifier);
		BaseFloatVectorWriter ivector_writer(new_ivector_specifier);
		Int32Writer num_utts_writer(new_num_utts_specifier);
		Vector<BaseFloat> processed_ivector;
		Vector<BaseFloat> spk_sum;
		int32 utt_count = 0;
		int32 num_utt_spk = 0;
		if (num_utts_reader.HasKey(spkid)) { // Check if the given speaker id exists, if so, update the existing speaker model
			for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) { // read through all registered speaker model
				std::string spk_model = train_ivector_reader.Key(); // get current speaker id
				if (spk_model == spkid) { // if the current the speaker is the one we need to update
					std::cout << "Found registered speaker: " << spkid << ". Update speaker model\n";
					spk_sum = train_ivector_reader.Value(); // get the current sum ivector of the speaker
					spk_sum.AddVec(1.0, *ivector); // add the new ivector to the sum
					num_utt_spk = num_utts_reader.Value(spkid); // get the current # of speech of the speaker
					utt_count = 1 + num_utt_spk; // # of speech + 1			
					ivector_writer.Write(spk_model, spk_sum); 
					num_utts_writer.Write(spk_model, utt_count); // re-write
				}
				else { //if the current the speaker is NOT the one we need to update
					num_utt_spk = num_utts_reader.Value(spk_model);
					const Vector<BaseFloat> &train_ivector = train_ivector_reader.Value();
					ivector_writer.Write(spk_model, train_ivector);
					num_utts_writer.Write(spk_model, num_utt_spk); // read and re-rewrite, keep looping to next registered speaker
				}
				
			}
		} else { // If there speaker id is new
			std::cout << "No registered speaker: " << spkid << ". Create a new spkid\n";
			utt_count++;
			ivector_writer.Write(spkid, *ivector);
			num_utts_writer.Write(spkid, utt_count); // write the new speaker 
			for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
				std::string spk_model = train_ivector_reader.Key();
				const Vector<BaseFloat> &train_ivector = train_ivector_reader.Value();
				num_utt_spk = num_utts_reader.Value(spk_model);
				ivector_writer.Write(spk_model, train_ivector);
				num_utts_writer.Write(spk_model, num_utt_spk); // add the existed registered models
			}
		}
	train_ivector_reader.Close();
	num_utts_reader.Close();
	ivector_writer.Close();
	num_utts_writer.Close();
	}
	
	// Training process
	void TrainProcess(std::string spkid, Matrix<BaseFloat> *feats) {
		Matrix<BaseFloat> processed_feats;
		MfccProcess(feats, &processed_feats); // Append deltas and double deltas, apply CMVN
		Posterior post;
		SelectPost(&processed_feats, &post); // Precompute Gaussian indices
		Vector<BaseFloat> ivector;
		ExtractIvectors(&processed_feats, &post, &ivector); // Extract i-vector
		UpdateSpkModel(spkid, &ivector); // Update/Create speaker model for speaker id
	}
}  // end namespace SpeakerIdentification

using namespace SpeakerIdentification;

int main(int argc, char *argv[]) {
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;
	// Read pre-trained mathmatics
	std::string model_dubm = "mat//final.dubm",
		model_ubm = "mat//final.ubm",
		ivector_extractor = "mat//final.ie",
		matrix_rxfilename = "mat//transform.mat",
		mean_rxfilename = "mat//mean_lw.vec";

	ReadKaldiObject(model_dubm, &gmm);
	ReadKaldiObject(model_ubm, &fgmm);
	ReadKaldiObject(ivector_extractor, &extractor);
	ReadKaldiObject(matrix_rxfilename, &transform);
	ReadKaldiObject(mean_rxfilename, &mean);

	// Read input
	if (argc != 3){
		std::cerr << "Usage: enroll spkid speech.wav!\n";
	}
	std::string spkid = argv[1]; // speaker id
	std::string train_speech = argv[2]; // speech waveform
	std::ifstream is(train_speech, std::ios_base::binary); // test input, 16s speech
	WaveData wave;
	wave.Read(is); // read speech waveform
	KALDI_ASSERT(wave.Data().NumRows() == 1);
	SubVector<BaseFloat> waveform(wave.Data(), 0);
	Vector<BaseFloat> waveform_vad;
	//waveform_vad.CopyFromVec(waveform);
	Vector<BaseFloat> wave_piece_buffer;
	std::string old_iv = "iv//train_iv.ark",
		old_num_utts = "iv//train_num_utts.ark",
		new_iv_scp = "iv//train_iv_new.scp",
		new_iv = "iv//train_iv_new.ark",
		new_num_utts = "iv//train_num_utts_new.ark";
     	
	std::ifstream ifile(old_iv);
	if (!ifile) { // create model file if none exists
  		std::string ivector_specifier = "ark:iv//train_iv.ark",
			    num_utts_specifier = "ark,t:iv//train_num_utts.ark";
		BaseFloatVectorWriter ivector_writer(ivector_specifier);
		Int32Writer num_utts_writer(num_utts_specifier);
	}
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
	std::cout << "The total active speech is " << waveform_vad.Dim()/wave.SampFreq() << " seconds.\n";
	// the parametrization object
	MfccOptions op_mfcc;
	MfccInitiation(&wave, &op_mfcc); // initialize MFCC options based on the audio sampling rate
	Mfcc mfcc(op_mfcc);
	Matrix<BaseFloat> features;
	mfcc.Compute(waveform_vad, 1.0, &features); // compute MFCC features of the speech
	TrainProcess(spkid, &features); // (if no VAD) Training process

	if( remove( old_iv.c_str() ) != 0 )
		std::cout << "Failed to delete '" << old_iv << "': " << strerror(errno) << '\n';
	if( remove( old_num_utts.c_str() ) != 0 )
		std::cout << "Failed to delete '" << old_num_utts << "': " << strerror(errno) << '\n';
	if( remove( new_iv_scp.c_str() ) != 0 )
		std::cout << "Failed to delete '" << new_iv_scp << "': " << strerror(errno) << '\n';
  	if ( rename( new_iv.c_str() , old_iv.c_str() )!=0)
		std::cout << "Failed to rename '" << new_iv << "': " << strerror(errno) << '\n';
	if ( rename( new_num_utts.c_str() , old_num_utts.c_str() )!=0)
		std::cout << "Failed to rename '" << new_num_utts << "': " << strerror(errno) << '\n'; // Deleting and Renaming speaker model files.
	std::cout << "Done.\n";
}
