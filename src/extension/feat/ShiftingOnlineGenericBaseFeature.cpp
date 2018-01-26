template<class C>
void ShiftingOnlineGenericBaseFeature<C>::GetFrame(int32 frame, VectorBase<BaseFloat> *feat)
{
 // 'at' does size checking. 
 feat->CopyFromVec(*(features_.at(frame)));
};

template<class C>
ShiftingOnlineGenericBaseFeature<C>::ShiftingOnlineGenericBaseFeature(const typename C::Options &opts):
 computer_(opts),
 window_function_(computer_.GetFrameOptions()),
 input_finished_(false),
 new_segment_(false),
 waveform_offset_(0)
{ }

template<class C>
void ShiftingOnlineGenericBaseFeature<C>::AcceptWaveform(BaseFloat sampling_rate, const VectorBase<BaseFloat> &waveform, bool vad)
{
 BaseFloat expected_sampling_rate = computer_.GetFrameOptions().samp_freq;
 if (sampling_rate != expected_sampling_rate)
   KALDI_ERR << "Sampling frequency mismatch, expected "
             << expected_sampling_rate << ", got " << sampling_rate;
 if (waveform.Dim() == 0)
   return;  // Nothing to do.
 //if (input_finished_)
  // KALDI_ERR << "AcceptWaveform called after InputFinished() was called.";
 // append 'waveform' to 'waveform_remainder_.'
 Vector<BaseFloat> appended_wave(waveform_remainder_.Dim() + waveform.Dim());
 if (waveform_remainder_.Dim() != 0)
   appended_wave.Range(0, waveform_remainder_.Dim()).CopyFromVec(
       waveform_remainder_);
 appended_wave.Range(waveform_remainder_.Dim(), waveform.Dim()).CopyFromVec(
     waveform);
 waveform_remainder_.Swap(&appended_wave);
 ComputeFeatures(vad);
}

template<class C>
void ShiftingOnlineGenericBaseFeature<C>::NewSegment(ShiftingOnlineFeatureInterface *a, int32 segment_shift_)
{
 new_segment_ = true;
 int32 dim_ = a->Dim();
 int32 nFrame_ = a->NumFramesReady();
 int32 nFrame_remain_ = nFrame_ - segment_shift_;
 //OnlineCacheFeature cache_(a);
 std::vector<Vector<BaseFloat>* > new_features_;
 for (int32 frame_num = segment_shift_; frame_num < nFrame_; frame_num++) {
  Vector<BaseFloat> garbage(dim_);
  a->GetFrame(frame_num, &garbage);
  new_features_.push_back(new Vector<BaseFloat>(garbage));
 }
 KALDI_ASSERT(new_features_.size() == nFrame_remain_);
 //cache_.ClearCache();  
 DeletePointers(&features_);
 features_.resize(nFrame_remain_);
 for (int32 frame_num = 0; frame_num < nFrame_remain_; frame_num++) {
  features_[frame_num] = new_features_[frame_num];
 }
 new_features_.clear();
 waveform_offset_ = nFrame_remain_ * 80;
 vad_result_.clear();
}

template<class C>
void ShiftingOnlineGenericBaseFeature<C>::ComputeFeatures(bool vad)
{
 const FrameExtractionOptions &frame_opts = computer_.GetFrameOptions();
 int64 num_samples_total = waveform_offset_ + waveform_remainder_.Dim();
 int32 num_frames_old = features_.size(),
     num_frames_new = NumFrames(num_samples_total, frame_opts,
                                input_finished_);
 KALDI_ASSERT(num_frames_new >= num_frames_old);
 features_.resize(num_frames_new, NULL);
 if (vad == true){

  vad_result_.resize(num_frames_new, 1);
  Vector<BaseFloat> window;
  bool need_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 frame = num_frames_old; frame < num_frames_new; frame++) {
   BaseFloat raw_log_energy = 0.0;
   ExtractWindow(waveform_offset_, waveform_remainder_, frame,
                 frame_opts, window_function_, &window,
                 need_raw_log_energy ? &raw_log_energy : NULL);
   Vector<BaseFloat> *this_feature = new Vector<BaseFloat>(computer_.Dim(),
                                                           kUndefined);
   // note: this online feature-extraction code does not support VTLN.
   BaseFloat vtln_warp = 1.0;
   computer_.Compute(raw_log_energy, vtln_warp, &window, this_feature);
   features_[frame] = this_feature;
  }
 }else{
  for (int32 frame = num_frames_old; frame < num_frames_new; frame++) {
   Vector<BaseFloat> *this_feature = new Vector<BaseFloat>(computer_.Dim(), kSetZero);
   features_[frame] = this_feature;
  }
  vad_result_.resize(num_frames_new, 0);
 }
 // OK, we will now discard any portion of the signal that will not be
 // necessary to compute frames in the future.
 int64 first_sample_of_next_frame = FirstSampleOfFrame(num_frames_new,
                                                       frame_opts);
 int32 samples_to_discard = first_sample_of_next_frame - waveform_offset_;
 if (samples_to_discard > 0) {
   // discard the leftmost part of the waveform that we no longer need.
   int32 new_num_samples = waveform_remainder_.Dim() - samples_to_discard;
   if (new_num_samples <= 0) {
     // odd, but we'll try to handle it.
     waveform_offset_ += waveform_remainder_.Dim();
     waveform_remainder_.Resize(0);
   } else {
     Vector<BaseFloat> new_remainder(new_num_samples);
     new_remainder.CopyFromVec(waveform_remainder_.Range(samples_to_discard,
                                                         new_num_samples));
     waveform_offset_ += samples_to_discard;
     waveform_remainder_.Swap(&new_remainder);
   }
 }
}

//template<class C>
void ShiftingOnlineCacheFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(frame >= 0);
  if (static_cast<size_t>(frame) < cache_.size() && cache_[frame] != NULL) {
    feat->CopyFromVec(*(cache_[frame]));
  } else {
    if (static_cast<size_t>(frame) >= cache_.size())      
      cache_.resize(frame + 1, NULL);
    int32 dim = this->Dim();
    cache_[frame] = new Vector<BaseFloat>(dim); 
    // The following call will crash if frame "frame" is not ready.
    src_->GetFrame(frame, cache_[frame]);
    
    feat->CopyFromVec(*(cache_[frame]));
  }
}

//template<class C>
void ShiftingOnlineCacheFeature::ClearCache() {
  for (size_t i = 0; i < cache_.size(); i++)
    delete cache_[i];
  cache_.resize(0);
}
