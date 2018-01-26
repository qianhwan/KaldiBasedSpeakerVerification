// Redefining shiftingonline-feature-itf template and adding vad feature
#ifndef SHIFTING_ONLINE_FEATURE_ITF_H_
#define SHIFTING_ONLINE_FEATURE_ITF_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

class ShiftingOnlineFeatureInterface {
 public:
  virtual int32 Dim() const = 0; /// returns the feature dimension.
  
  /// Returns the total number of frames, since the start of the utterance, that
  /// are now available.  In an online-decoding context, this will likely
  /// increase with time as more data becomes available.
  virtual int32 NumFramesReady() const = 0;
  
  /// Returns true if this is the last frame.  Frame indices are zero-based, so the
  /// first frame is zero.  IsLastFrame(-1) will return false, unless the file
  /// is empty (which is a case that I'm not sure all the code will handle, so
  /// be careful).  This function may return false for some frame if
  /// we haven't yet decided to terminate decoding, but later true if we decide
  /// to terminate decoding.  This function exists mainly to correctly handle
  /// end effects in feature extraction, and is not a mechanism to determine how
  /// many frames are in the decodable object (as it used to be, and for backward
  /// compatibility, still is, in the Decodable interface).
  virtual bool IsLastFrame(int32 frame) const = 0;
  virtual int GetVad(int frame) = 0;
  virtual int NumVadReady() const = 0;
  /// Gets the feature vector for this frame.  Before calling this for a given
  /// frame, it is assumed that you called NumFramesReady() and it returned a
  /// number greater than "frame".  Otherwise this call will likely crash with
  /// an assert failure.  This function is not declared const, in case there is
  /// some kind of caching going on, but most of the time it shouldn't modify
  /// the class.
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) = 0;
  
  // Returns frame shift in seconds.  Helps to estimate duration from frame
  // counts.
  virtual BaseFloat FrameShiftInSeconds() const = 0;

  /// Virtual destructor.  Note: constructors that take another member of
  /// type OnlineFeatureInterface are not expected to take ownership of
  /// that pointer; the caller needs to keep track of that manually.
  virtual ~ShiftingOnlineFeatureInterface() { }  
  
};


/// Add a virtual class for "source" features such as MFCC or PLP or pitch
/// features.
class ShiftingOnlineBaseFeature: public ShiftingOnlineFeatureInterface {
 public:
  /// This would be called from the application, when you get more wave data.
  /// Note: the sampling_rate is typically only provided so the code can assert
  /// that it matches the sampling rate expected in the options.
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform, bool vad) = 0;

  /// InputFinished() tells the class you won't be providing any
  /// more waveform.  This will help flush out the last few frames
  /// of delta or LDA features (it will typically affect the return value
  /// of IsLastFrame.
  virtual void InputFinished() = 0;

};


/// @}
}  // namespace Kaldi

#endif  // KALDI_SHIFTING_ONLINE_FEATURE_ITF_H_
