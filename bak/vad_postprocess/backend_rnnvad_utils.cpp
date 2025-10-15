#include "backend/mr527/backend_rnnvad_utils.h"
#define RTCVAD_MODE 3
#define RTCSR 16000
// RNNVadFulfill::RNNVadFulfill():RNNVadFulfill(int vad_type) {}
#include <iostream>
RNNVadFulfill::RNNVadFulfill(int vad_type, int language_type) {
    type_ = vad_type;
    language_ = language_type;
    std::cout << "create vad type : " << type_ << std::endl;
    if(vad_type == 0)
    {
      vadobj = new RRVAD::RNNVADOBJ;
      vadobj->pvad = RRVAD::rnnoise_create();
      RRVAD::rnnoise_init(vadobj->pvad);
    }
    else if(vad_type == 1)
    {
      fvad_obj = RRVAD_F::fvad_new();
      RRVAD_F::fvad_set_mode(fvad_obj,RTCVAD_MODE);
      RRVAD_F::fvad_set_sample_rate(fvad_obj,RTCSR);
    }
    else
    {
      std::cerr << "Error: Unexpected VAD type\n"; 
      assert(0);
    }
}


RNNVAD_RESULTS RNNVadFulfill::VadProcessFrame(float* frame, float& vad_prob){
    if(type_ == 0)
    {
      vad_prob = RRVAD::rnnoise_process_frame(vadobj->pvad, frame, frame, 1);
      // std::cout << "      0     " << vad_prob << std::endl;

    }
    else if(type_ == 1)
    {
      int16_t rtcvad_in[RNNVAD_FRAME_SIZE] = {0};
      for(int k=0;k<RNNVAD_FRAME_SIZE;++k)
      {
        rtcvad_in[k] = static_cast<int16_t>(*(frame + k));
      }
      vad_prob = static_cast<float>( RRVAD_F::fvad_process(fvad_obj, rtcvad_in, RNNVAD_FRAME_SIZE));
      // std::cout << "     1      " << vad_prob << std::endl;
    }
    else
    {
      std::cerr << "Error: Unexpected VAD type\n"; 
      assert(0);
    }
    RNNVAD_RESULTS res = post.do_postprocess(vad_prob, language_);
    return res;
}

void RNNVadFulfill::RNNVadDestroy(){
  if(type_ == 0)
  {
    RRVAD::rnnoise_destroy(vadobj->pvad);
    delete vadobj;
  }
  else if(type_ == 1)
  {
    RRVAD_F::fvad_free(fvad_obj);
  }
  else
  {
    std::cerr << "Error: Unexpected VAD type\n"; 
    assert(0);
  }
}
