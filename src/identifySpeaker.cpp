// identifySpeaker.cpp
// Online testing for speaker identification
// QIANHUI WAN, University of Ottawa
// Jan-24-2018
//
// Usage: identifySpeaker speech.wav
// Given an audio contains speech, this code will read the audio every 10ms, and make a speaker verification
// decision on 800 frame windows with 320 frame window shift.
 
// The following codes create two thread for processing, Thread_Read abd Thread_Process, both threads are 
// put to sleep in most time.
// Thread_Read will collect 10ms audio segments every 10ms and extract features (MFCC) from the audio segments, 
// it will be notified by newAudio_ready flag.
// Thread_Process will process a 800-frame feature buffer and make decision based on these 800-frame feature buffer, 
// it will be notified by newTest_ready flag.

#include <time.h>
#include <string>

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include "feat/online-feature.h"
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

#include "feat/ShiftingOnlineGenericBaseFeature.h"
extern "C"
{
  #include <fvad.h>
}

using namespace kaldi;
using namespace std;


namespace SpeakerIdentification {
	// lock and global var shared between threads
 	pthread_mutex_t m_wave = PTHREAD_MUTEX_INITIALIZER;
 	pthread_mutex_t m_mfcc = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t cv_wave = PTHREAD_COND_INITIALIZER;
 	pthread_cond_t cv_mfcc = PTHREAD_COND_INITIALIZER;
 	// notification flags between threads
 	bool newAudio_ready = false;
 	bool newTest_ready = false;
 	bool file_finish = false;
 	bool enough_vad = false;
 	// Set parameters
 	int32 segment_shift = 320; // segment shift, a new decision is made every 320 frames (~ 3.2 seconds)
 	int32 segment_size = 800; //segment length, decisions are made based on 800 frames (~ 8 seconds)
 	float accept_threshold = 15.0; // threshold, can be adjust due to different system sensitivity
 	Fvad *vad = NULL; // create vad
 	bool vadres;
 	BaseFloat score;
 	int sampling_rate;
 	// Initialize audio and feature buffers
 	Matrix<BaseFloat> mfcc_feats_buffer(segment_size, 20, kSetZero); // Feature buffer (800 * 20) initalized with zeros, 
                                                                         // 20 is the dimension of MFCC
 	Matrix<BaseFloat> mfcc_selected_buffer;
 	Vector<BaseFloat> wave_piece_buffer; // Audio buffer, store 10ms audio
 	vector<int> vad_result_flag(3, 0);
 	vector<int> vad_results_buffer(segment_size, 0);
 	// Initialize parameters and pre-trained models
 	MfccOptions op_mfcc; // MFCC parameters
 	DiagGmm gmm; // Diagonal covariance UBM, pre-trained model
 	FullGmm fgmm; // Full covariance UBM, pre-trained model
 	IvectorExtractor extractor; // I-vector extractor, pre-trained model
 	Matrix<BaseFloat> transform; // LDA + WCCN compensation model, pre-trained model
 	Vector<BaseFloat> mean; // Global mean of i-vectors, pre-trained model
 	Plda plda; // PLDA, pre-trained model

 	bool process_vad(Vector<BaseFloat> wave_piece_buffer, Fvad *vad, size_t framelen){
    
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

 	// bring the last two flags to the front, and add the new vad result to the last index
 	void UpdateVadFlag(vector<int> *vad_result_flag, bool vadres){
  		if (vad_result_flag->size() != 3)
   			std::cerr << "wrong dimesion of vad_result_flag!\n";
  		vector<int> temp(vad_result_flag->size(), 0);
  		for (int i = 0; i < vad_result_flag->size() - 1; i++){
   			temp[i] = (*vad_result_flag)[i+1];
  		}
  		temp[2] = vadres;
  		*vad_result_flag = temp;
 	}

 	// Copy MFCC features from OnlineFeatureInterface* to Matrix<BaseFloat>*
 	void GetSpeechMfcc(ShiftingOnlineFeatureInterface *a, Matrix<BaseFloat> *output, vector<int> *vad_result_buffer) {  
  		int32 dim = a->Dim();
  		int32 nFrame = a->NumFramesReady();
  		KALDI_ASSERT(segment_shift == nFrame);
  		ShiftingOnlineCacheFeature cache(a);
  		std::vector<Vector<BaseFloat>* > cached_frames;
  		for (int32 frame_num = 0; frame_num < nFrame; frame_num++) {
   			Vector<BaseFloat> garbage(dim);
   			cache.GetFrame(frame_num, &garbage);
   			cached_frames.push_back(new Vector<BaseFloat>(garbage));
   			(*vad_result_buffer)[frame_num + segment_size - segment_shift] = cache.GetVad(frame_num);
  		}
  		KALDI_ASSERT(cached_frames.size() == segment_shift);
  
  		for (int32 i = 0; i < cached_frames.size(); i++) {
   			output->CopyRowFromVec(*(cached_frames[i]), i + segment_size - segment_shift);
   			delete cached_frames[i];
  		}
  		cached_frames.clear();
  		cache.ClearCache();
 	}

 	// Update mfcc_feats_buffer(800), discard the first 320 frames and move the last 480 frames to the beginning of buffer.
 	// The last 320 frames of updated buffer are 0s
 	void UpdateMFCCBuffer(Matrix<BaseFloat> *feats, vector<int> *vad_result_buffer) {
  		Matrix<BaseFloat> new_feats(feats->NumRows(), feats->NumCols(), kSetZero);
  		Vector<BaseFloat> cache(feats->NumCols());
  		vector<int> vad_temp(vad_result_buffer->size(), 0);
  		for (int32 i = 0; i < segment_size - segment_shift; i++) {
   			cache = feats->Row(i + segment_shift);
   			new_feats.CopyRowFromVec(cache, i);
   			cache.Resize(feats->NumCols(), kSetZero);
   			vad_temp[i] = (*vad_result_buffer)[i + segment_shift];
  		}
  		*feats = new_feats;
  		*vad_result_buffer = vad_temp;
 	}

 	void SelectVadFrames(Matrix<BaseFloat> *feats, vector<int> *vad_result_buffer, Matrix<BaseFloat> *mfcc_selected_buffer){
  		int32 num_frame = 0;
  		KALDI_ASSERT(feats->NumRows() == vad_result_buffer->size());
  		int32 dim = feats->NumCols();
		std::vector<Vector<BaseFloat>* > selected_frames;
  
  		for (int32 i = 0; i < vad_result_buffer->size(); i++){
   			if ((*vad_result_buffer)[i] == 1){
     				Vector<BaseFloat> temp(dim);
     				temp = feats->Row(i);
     				selected_frames.push_back(new Vector<BaseFloat>(temp));
   			}		
  		}
  		if (selected_frames.size() < 200){
    			std::cout << "Effective Speech shorter than 2s , skip verifying.\n";
  		}else{
    			std::cout << "Effective speech length: "<< (selected_frames.size() - 1) * 0.01 + 0.025 << "s.";
    			enough_vad = true;
    			mfcc_selected_buffer->Resize(selected_frames.size(), dim);
    			for (int32 i = 0; i < selected_frames.size(); i++) {
      				mfcc_selected_buffer->CopyRowFromVec(*(selected_frames[i]), i);
      				delete selected_frames[i];
    			}
  		}
  		selected_frames.clear();
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
  		// Applying CMVN
  		Matrix<BaseFloat> cmvn_feats_(delta_feats.NumRows(),
   		delta_feats.NumCols(), kUndefined);
  		SlidingWindowCmn(cmvn_opts, delta_feats, &cmvn_feats_); // Apply sliding window CMVN
  		*processed_feats = cmvn_feats_;
 	}

 	// Select Precompute Gaussian indices for pruning
 	// For each frame, gives a list of the n best Gaussian indices sorted from best to worst
 	void SelectPost(Matrix<BaseFloat> *feats, Posterior *post) {
  		using std::vector;
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
  		double this_tot_loglike = 0;
  		bool utt_ok = true;
  		for (int32 t = 0; t < num_frames; t++) {
   			SubVector<BaseFloat> frame(feats_, t);
   			const std::vector<int32> &this_gselect = gselect[t];
   			KALDI_ASSERT(!gselect[t].empty());
   			Vector<BaseFloat> loglikes;
   			fgmm.LogLikelihoodsPreselect(frame, this_gselect, &loglikes);
   			this_tot_loglike += loglikes.ApplySoftMax();
   			if (fabs(loglikes.Sum() - 1.0) > 0.01) {
    				utt_ok = false;
   			}else {
    				if (min_post != 0.0) {
     					int32 max_index = 0; 
     					loglikes.Max(&max_index);
     					for (int32 i = 0; i < loglikes.Dim(); i++)
      						if (loglikes(i) < min_post)
       							loglikes(i) = 0.0;
    					BaseFloat sum = loglikes.Sum();
     					if (sum == 0.0) {
      						loglikes(max_index) = 1.0;
     					}else {
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
  		using std::vector;
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
  		double this_t = opts.acoustic_weight * TotalPosterior(*post),max_count_scale = 1.0;
  		if (opts.max_count > 0 && this_t > opts.max_count) {
   			max_count_scale = opts.max_count / this_t;
   			KALDI_LOG << "Scaling stats by scale "
    			<< max_count_scale << " due to --max-count="
    			<< opts.max_count;
   			this_t = opts.max_count;
  		}
  		ScalePosterior(opts.acoustic_weight * max_count_scale, post);
  		bool need_2nd_order_stats = false;
  		IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(), extractor.FeatDim(), need_2nd_order_stats); 
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

 	// Apply length normalization, LDA and WCCN to i-vectors
 	void IvectorProcess(Vector<BaseFloat> *ivector, Vector<BaseFloat> *processed_ivector) {
  		SubMatrix<BaseFloat> linear_term(transform, 0, transform.NumRows(), 0, transform.NumCols() - 1);
  		Vector<BaseFloat> constant_term(transform.NumRows());
  		constant_term.CopyColFromMat(transform, transform.NumCols() - 1);
  		Vector<double> sum(transform.NumRows());
  		double sumsq = 0.0;
  		BaseFloat norm = ivector->Norm(2.0);
  		BaseFloat ratio = norm / sqrt(ivector->Dim());
  		if (ratio == 0.0) {
   			KALDI_WARN << "Zero iVector";
  		}else {
   			ivector->Scale(1.0 / ratio);
  		}
  		Vector<BaseFloat> transformed_ivector(transform.NumRows());
  		if (ivector->Dim() == transform.NumCols()) {
   			transformed_ivector.AddMatVec(1.0, transform, kNoTrans, *ivector, 0.0);
  		}else {
   			KALDI_ASSERT(ivector->Dim() == transform.NumCols() - 1);
   			transformed_ivector.CopyFromVec(constant_term);
   			transformed_ivector.AddMatVec(1.0, linear_term, kNoTrans, *ivector, 1.0);
  		}
  		sum.AddVec(1.0, transformed_ivector);
  		sumsq += VecVec(transformed_ivector, transformed_ivector);
  		transformed_ivector.AddVec(-1.0, mean);
  		*processed_ivector = transformed_ivector;
 	}

 	// Compare a processed i-vector to all registered speaker models
 	void ComputeScore(Vector<BaseFloat> *ivector) {
  		typedef std::string string;
  		PldaConfig plda_config;
  		int32 dim = plda.Dim();

  		// Transforms an iVector into a space where the within-class variance is unit and between-class variance is diagonalized.
  		int32 num_examples = 1;
  		Vector<BaseFloat> *transformed_ivector_test = new Vector<BaseFloat>(dim);
  		BaseFloat score_tmp;
  		string speaker_id;
  		score=-1e9; // initialize to very small value
  
  		plda.TransformIvector(plda_config, *ivector, num_examples, transformed_ivector_test);
  		Vector<double>  test_ivector_dbl(*transformed_ivector_test);
  		// Read Registered speaker models, two things are stored for each registered speaker:
  		// 1. the sum of all i-vectors (one ivector from one speech) belong to him/her,
  		// 2. the number of speech for that speaker
  		// this to compute the average i-vector for each speaker later on
  		SequentialBaseFloatVectorReader train_ivector_reader("ark:iv//train_iv.ark"); 
  		RandomAccessInt32Reader num_utts_reader("ark:iv//train_num_utts.ark"); 
  		// Reading training ivectors, the reader will sort through all regestered speakers
  		for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
   			std::string spk = train_ivector_reader.Key(); // get current speaker id
   			Vector<BaseFloat> spk_sum = train_ivector_reader.Value(); // get the sum ivector of current speaker id
   			int32 num_train_examples;
   			if (!num_utts_reader.HasKey(spk)) {
    				KALDI_WARN << "Number of utterances not given for speaker " << spk; 
    				continue;
   			} // throw error if there is no information of the speech number of that speaker
   			num_train_examples = num_utts_reader.Value(spk); // get the speech number of current speaker id
   			spk_sum.Scale(1.0 / num_train_examples); // get the averaged i-vector for current speaker id
   			// Apply length normalization, LDA and WCCN to i-vectors
   			Vector<BaseFloat> spk_mean;
   			IvectorProcess(&spk_sum, &spk_mean);
   			// Transforms an iVector into a space where the within-class variance is unit and between-class variance is diagonalized.
   			const Vector<BaseFloat> &train_ivector = spk_mean;
   			Vector<BaseFloat> *transformed_ivector_train = new Vector<BaseFloat>(dim);
   			plda.TransformIvector(plda_config, train_ivector, num_train_examples, transformed_ivector_train);
   			Vector<double> train_ivector_dbl(*transformed_ivector_train);
   			// Compute PLDA scoring for all speaker models (even if a match with score >= threshold is found), 
   			// to get the highest possible score (helps evaluating the performance, adjusting threshold, etc.)
   			score_tmp = plda.LogLikelihoodRatio(train_ivector_dbl, num_train_examples, test_ivector_dbl);  
   			if (score_tmp > score){
	   			score=score_tmp;
	   			speaker_id= spk;
   			}
  		}
  		train_ivector_reader.Close();
  		if (score >= accept_threshold){
    			std::cout << "Family member detected! Speaker: " << speaker_id << "\t(score: " << score << ")" << std::endl;
   		}else {   // If none of the registered speaker model gives a score that is higher than the threshold
   			std::cout << "No family member detected." << "\t\t(score: " << score << ")" << std::endl;
  		}
 	}

 	// Process a 800 frame feature buffer
 	void TestProcess(Matrix<BaseFloat> *feats) {
  		Matrix<BaseFloat> processed_feats;
  		MfccProcess(feats, &processed_feats); // Append deltas and double deltas, apply CMVN
		Posterior post;
  		SelectPost(&processed_feats, &post); // Precompute Gaussian indices
  		Vector<BaseFloat> ivector;
  		ExtractIvectors(&processed_feats, &post, &ivector); // Extract i-vector
  		Vector<BaseFloat> processed_ivector;
  		IvectorProcess(&ivector, &processed_ivector); // Apply i-vector length normalizatin, LDA and WCCN to i-vector
  		ComputeScore(&processed_ivector); // Compare the processed i-vector to all registered speaker models
 	}

 	// Thread_Process, processing 800-frame (8 seconds) feature buffer every 3.2 seconds
 	void* Thread_Process(void*){
  	// Read pre-trained background models
  		std::string model_dubm = "mat//final.dubm",
   		model_ubm = "mat//final.ubm",
   		ivector_extractor = "mat//final.ie",
   		matrix_rxfilename = "mat//transform.mat",
   		mean_rxfilename = "mat//mean_lw.vec",
   		plda_rxfilename = "mat//final.plda";
   
  		ReadKaldiObject(model_dubm, &gmm);
  		ReadKaldiObject(model_ubm, &fgmm);
  		ReadKaldiObject(ivector_extractor, &extractor);
  		ReadKaldiObject(matrix_rxfilename, &transform);
  		ReadKaldiObject(mean_rxfilename, &mean);
  		ReadKaldiObject(plda_rxfilename, &plda);
 		// sleep and wake-up
		for(;;){
			// Wait until Thread_Audio sends notification: newTest_ready = true
			// lock mutex, check spurious wakeup
			pthread_mutex_lock( &m_mfcc );
			while(!newTest_ready&&!file_finish)
	      			pthread_cond_wait( &cv_mfcc, &m_mfcc );
			if(file_finish) return NULL; // Finish thread if entire file is finished
			// Do test process on mfcc_feats_buffer
  			SelectVadFrames(&mfcc_feats_buffer, &vad_results_buffer, &mfcc_selected_buffer);
			if (enough_vad == true)
				TestProcess(&mfcc_selected_buffer);
			mfcc_selected_buffer.Resize(0, 0);
			enough_vad = false;
			UpdateMFCCBuffer(&mfcc_feats_buffer, &vad_results_buffer); // Update mfcc_feats_buffer, 
                                                                                   // discard old 320 frames and move the left 480 to the top
			newTest_ready = false; // Reset newTest_ready flag to false
			pthread_mutex_unlock( &m_mfcc ); // Unlock mutex
																		   
		 } 
	}


 	// Thread_Read, read 10ms audio segments and extract MFCCs 
	void* Thread_Read(void*)
	{   
		// online_mfcc is a MFCC online extractor, it has two buffers inside:
		// 1. wavefrom_remainder, a short piece of waveform that we may need to keep
		// after extracting all the whole frames we can (whatever length of feature
		// will be required for the next phase of computation).
		// 2. features, contains all the MFCCs we have extracted
		ShiftingOnlineGenericBaseFeature<MfccComputer> online_mfcc(op_mfcc);
		for(;;){
			pthread_mutex_lock( &m_wave ); // lock mutex 
			while(!newAudio_ready&&!file_finish) // check spurious wakeup
      				pthread_cond_wait( &cv_wave, &m_wave );
			if(file_finish){
				online_mfcc.InputFinished();
				return NULL;
			} // Finish thread if entire file is finished
			bool flag = true;
			for (int i = 0; i < vad_result_flag.size(); i++){
				if (vad_result_flag[i] == 0){
					flag = false;
					break;
				}
			}	
			online_mfcc.AcceptWaveform(sampling_rate, wave_piece_buffer, flag); // calculate MFCCs with given wave_piece_buffer(10ms)		
			int32 num_frames = online_mfcc.NumFramesReady(); // get the number of frames of online_mfcc
			if (num_frames >= segment_shift) { // Check if there is frames ready in online_mfcc, if so
				pthread_mutex_lock( &m_mfcc ); // lock mutex
				GetSpeechMfcc(&online_mfcc, &mfcc_feats_buffer, &vad_results_buffer); // copy the features in online_mfcc to mfcc_feats_buffer(800)
				newTest_ready = true; // set new_Test flag to true
				pthread_mutex_unlock( &m_mfcc ); // unlock mutex
				pthread_cond_signal( &cv_mfcc ); // notify Thread_Process
				online_mfcc.NewSegment(&online_mfcc, online_mfcc.NumFramesReady()); // Reset online_mfcc, discard all 320 frames.
			}
			newAudio_ready = false; // Reset newAudio flag to false
			pthread_mutex_unlock( &m_wave ); // unlock mutex
		}
	}

}  // end namespace SpeakerIdentification

using namespace SpeakerIdentification;

bool fexists(const char *filename){
  	ifstream ifile(filename);
  	return ifile;
}

int main(int argc, char *argv[]){
 	// First of all, check if /iv and its contents exist
	if(!fexists("iv//train_iv.ark") || !fexists("iv//train_num_utts.ark"))
		std::cerr << "No registered speaker info, enroll speaker first!\n";

	// Read input
 	if (argc != 2){
  		std::cerr << "Usage: identifySpeaker speech.wav!\n";
 	}
 	std::string test_speech = argv[1]; // get input audio
 	std::ifstream is(test_speech, std::ios_base::binary); 
 	WaveData wave;
 	wave.Read(is); // Read input audio
 	KALDI_ASSERT(wave.Data().NumRows() == 1); // throw error is the input data is not one channel
 	MfccInitiation(&wave, &op_mfcc); // initialize MFCC options based on the audio sampling rate
 	SubVector<BaseFloat> waveform(wave.Data(), 0);
 	sampling_rate = wave.SampFreq();
 	int32 wave_length = waveform.Dim(), // audio length
 	audio_segment = 0.01 * wave.SampFreq(), // # of samples for 10ms
 	num_segment = wave_length / audio_segment, // # of 10ms segments in total
 	offset_start = 0;
 	size_t framelen = audio_segment; // vad configuration

 	vad = fvad_new();
 	int mode = 3; // vad mode
 	if (fvad_set_sample_rate(vad, wave.SampFreq()) < 0) { // vad configuration
  		std::cerr <<"invalid sample rate: "<< wave.SampFreq() << " Hz\n";
 	}
 	if (fvad_set_mode(vad, mode) < 0) { // vad configuration
  		std::cerr <<"invalid mode: "<< mode << "\n";
 	}
 	pthread_t thread1, thread2; 
 	pthread_create( &thread1, NULL, &Thread_Read, NULL);
 	pthread_create( &thread2, NULL, &Thread_Process, NULL); // create threads

 	// This is to simulate the case in real life, receiving 10ms audio segments every 10ms
 	for (int32 i = 0; i < num_segment; i++) {
  		pthread_mutex_lock( &m_wave ); // lock mutex
  		wave_piece_buffer = waveform.Range(offset_start, audio_segment);// fill wave_piece_buffer with new 10ms 
  		vadres = process_vad(wave_piece_buffer, vad, framelen); // do vad, discard if vad returns false
  		UpdateVadFlag(&vad_result_flag, vadres);
  		newAudio_ready = true; // set newAudio flag to true
  		pthread_mutex_unlock( &m_wave ); // unlock mutex
  		pthread_cond_signal( &cv_wave ); // notify Thread_Read
  		offset_start += audio_segment;
  		Sleep(0.01);// sleep for 10ms
 	}
 	file_finish = true; // set file_finish flag to true
 	pthread_cond_signal( &cv_wave ); 
 	pthread_cond_signal( &cv_mfcc );// notify both thread with file_finish = true to finish both threads
 
 	pthread_join( thread1, NULL);
 	pthread_join( thread2, NULL);
 	std::cout << "Speech data is finished!\n";
 	std::cout << "Done.\n";
}

