#include "rnnoise.h"
#include "backend/mr527/backend_rnnvad_postprocess.h"
#include "fvad.h"
// namespace speechengine {

// namespace rnnvad {

  class RNNVadFulfill {
    public:
      // RNNVadFulfill();
      RNNVadFulfill(int vad_type, int language_type);
      RNNVAD_RESULTS VadProcessFrame(float* frame, float& vad_prob);
      void RNNVadDestroy();
    private:
      int type_; /* 0 rnnvad , 1 webrtc vad */
      int language_;
      RRVAD::RNNVADOBJ* vadobj;
      RNNVAD_POSTPROCESS post;
      RRVAD_F::Fvad* fvad_obj;
  };
// }
// }