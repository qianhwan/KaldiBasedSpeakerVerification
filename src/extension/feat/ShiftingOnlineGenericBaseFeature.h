// Redefining OnlineGenericBaseFeature template and adding shifting feature
#ifndef SHIFTING_ONLINE_GENERIC_BASE_FEATURE_H_
#define SHIFTING_ONLINE_GENERIC_BASE_FEATURE_H_

#include <string>
#include <vector>
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/feature-functions.h"
#include "feat/ShiftingOnlineFeatureItf.h"

using namespace kaldi;

template<class C>
class ShiftingOnlineGenericBaseFeature: public ShiftingOnlineBaseFeature
{
 public:
  // Constructor
  explicit ShiftingOnlineGenericBaseFeature(const typename C::Options &opts);
  ~ShiftingOnlineGenericBaseFeature()
  {
   DeletePointers(&features_);
  }

  virtual int32 Dim() const { return computer_.Dim(); }
  virtual bool IsLastFrame(int32 frame) const
  {
   return input_finished_ && frame == NumFramesReady() - 1;
  }
  virtual BaseFloat FrameShiftInSeconds() const
  {
   return computer_.GetFrameOptions().frame_shift_ms / 1000.0f;
  }
  virtual int32 NumFramesReady() const { return features_.size(); }
  virtual int NumVadReady() const { return vad_result_.size(); } 
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  void NewSegment(ShiftingOnlineFeatureInterface *a, int32 segment_shift_);
  virtual void AcceptWaveform(BaseFloat sampling_rate, const VectorBase<BaseFloat> &waveform, bool vad);
  virtual void InputFinished()
  {
   input_finished_ = true;
   ComputeFeatures(0);
  }
  virtual int GetVad(int frame){ return vad_result_[frame]; }


 private:
  void ComputeFeatures(bool vad);
  C computer_;
  FeatureWindowFunction window_function_;
  std::vector<Vector<BaseFloat>*> features_;
  bool input_finished_;
  bool new_segment_;
  BaseFloat sampling_frequency_;
  int64 waveform_offset_;
  Vector<BaseFloat> waveform_remainder_;
  std::vector<int> vad_result_;
};

class ShiftingOnlineMatrixFeature: public ShiftingOnlineFeatureInterface {
 public:
  /// Caution: this class maintains the const reference from the constructor, so
  /// don't let it go out of scope while this object exists.
  explicit ShiftingOnlineMatrixFeature(const MatrixBase<BaseFloat> &mat): mat_(mat) { }

  virtual int32 Dim() const { return mat_.NumCols(); }

  virtual BaseFloat FrameShiftInSeconds() const {
    return 0.01f;
  }

  virtual int32 NumFramesReady() const { return mat_.NumRows(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
    feat->CopyFromVec(mat_.Row(frame));
  }

  virtual bool IsLastFrame(int32 frame) const {
    return (frame + 1 == mat_.NumRows());
  }


 private:
  const MatrixBase<BaseFloat> &mat_;
};

class ShiftingOnlineCacheFeature: public ShiftingOnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual int NumVadReady() const { return src_->NumVadReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  
  virtual int GetVad(int frame){ return src_->GetVad(frame); }

  virtual ~ShiftingOnlineCacheFeature() { ClearCache(); }

  // Things that are not in the shared interface:

  void ClearCache();  // this should be called if you change the underlying
                      // features in some way.
  
  explicit ShiftingOnlineCacheFeature(ShiftingOnlineFeatureInterface *src): src_(src) { }
 private:

  ShiftingOnlineFeatureInterface *src_;  // Not owned here
  std::vector<Vector<BaseFloat>* > cache_;
};

#include "feat/ShiftingOnlineGenericBaseFeature.cpp"

#endif  // SHIFTING_ONLINE_GENERIC_BASE_FEATURE_H_
