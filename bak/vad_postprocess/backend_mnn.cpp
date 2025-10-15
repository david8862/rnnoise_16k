//
// Created by xiaobin on 10/8/19.
//
#include <iomanip>
#include <iostream>
#include <dlfcn.h>
#include <unistd.h>
#include <string_view>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include "auxiliary/stream_ext.h"
#include "backend/backend_mnn.h"

#ifdef MNN_WAKEWORD_MODEL_PRECISE
#include "backend/mnn/backend_mnn_precise_utils.h"
#endif
#ifdef MNN_WAKEWORD_MODEL_RESNET
#include "backend/mnn/backend_mnn_wuw_utils.h"
#endif

#ifdef MNN_ASR_MODEL_QUARTZNET
#include "backend/mnn/backend_mnn_quartznet_utils.h"
#include "backend/mnn/backend_quartznet_asr_decoder.h"
#endif
#ifdef MNN_INTENTSLOT_MODEL_CNNATTN
#include "backend/mnn/backend_mnn_cnnattention_utils.h"
#endif
#include "backend/mnn/backend_regex_utils.h"
#include "backend/mnn/backend_asrslu_utils.h"
#include <capability/capabilities.h>
#include "auxiliary/rdlogging.h"

#include "timer.h"
using namespace std;
using namespace rd;

#ifdef BACKEND_LITE
int FLAGSFST_v = 0;
#endif

namespace speechengine {

#ifdef INIT_LOCK
std::mutex MNNImpl::mtx;
#endif

MNNImpl::MNNImpl() : model_type_(BACKEND_MODEL_UNKNOWN), initialized_(false) {}

MNNImpl::~MNNImpl()
{
  if (initialized_) Deinitialize();
}

bool MNNImpl::Initialize(const BackendParameters& parameters)
{
  model_type_ = parameters.ModelType();

#ifdef MNN_WITH_SPEEX_AEC
  if (model_type_ == MNN_MODEL_SPEEX_AEC) {
    AECExtraConfigs* aec_cfgs = (AECExtraConfigs*)parameters.GetAECParameters();
    aec_objs_.echoCanceller_ = new RoboRockAEC::EchoCanceller(
        aec_cfgs->samplerate, aec_cfgs->num_near_end_chn,
        aec_cfgs->num_far_end_chn, aec_cfgs->num_near_end_start_chn,
        aec_cfgs->num_far_end_start_chn, aec_cfgs->num_blocks,
        aec_cfgs->hop_size, aec_cfgs->enable_echo_detect,
        aec_cfgs->echo_detect_threshold, aec_cfgs->echo_detect_smooth_chunk);
  }
#endif

#ifdef INIT_LOCK
  if (parameters.RuntimeType() == RUNTIME_GPU) {
    mtx.lock();
  }
#endif
  int num_of_models = parameters.ModelNum();
  if (model_type_ == MNN_MODEL_QUARTZNET ||
      model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
      model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
    //num_of_models indicates number of deep learning models, which doesn't include the model config and the ngram lm
    if (model_type_ == MNN_MODEL_QUARTZNET_REGEX)
      num_of_models = parameters.ModelNum() - 3;
    else
      num_of_models = parameters.ModelNum() - 2;
    asr_objs_.use_language_model =
        use_language_model[parameters.LanguageType()];
    asr_objs_.language = parameters.LanguageType();
  }
  else if (model_type_ == MNN_MODEL_RNNVAD) {
    num_of_models = 0;
    // std::cout << "GET VAD " << parameters.GetVADType() << std::endl;
    #ifndef BACKEND_LITE
    vad_objs_.RNNVad_ = new RNNVadFulfill(
        parameters.GetVADType(),
        parameters.LanguageType_VAD()); /* 0 rnnvad , 1 rtcvad */
    #endif
  }
  else if (model_type_ == MNN_MODEL_WUW_RESNET) {
    speechengine::rrwuw::getGlobalWUW_listener_params().reset_by_language(
        parameters.LanguageType_WUW());
    switch (parameters.LanguageType_WUW()) {
      case 0:  // nihaoshitou
        speechengine::rrwuw::getGlobalWUW_listener_params()
            .pWuwProcessor.reset_param(1); /* 0 0.3, 1 0.5, 2 0.8 3 1.0*/
        break;
      case 1:  // hello rocky
        speechengine::rrwuw::getGlobalWUW_listener_params()
            .pWuwProcessor.reset_param(2); /* 0 0.3, 1 0.5, 2 0.8 3 1.0*/
        break;
      case 2:
        speechengine::rrwuw::getGlobalWUW_listener_params()
            .pWuwProcessor.reset_param(0); /* 0 0.3, 1 0.5, 2 0.8 3 1.0*/
        break;
      default:
        speechengine::rrwuw::getGlobalWUW_listener_params()
            .pWuwProcessor.reset_param(0);
        break;
    }
  }
#ifdef MNN_WITH_SIGNAL_DOA
  if (model_type_ == MNN_MODEL_SIGNAL_DOA) {
    #ifndef BACKEND_LITE
    doa_objs_.doa_obj_ = new rrse_SignalDOA::DOAEstimator();
    #endif
    num_of_models = 0;
  }
#endif
  for (int i = 0; i < num_of_models; i++) {
    assert(!parameters.HasModelStructure(i));

    MNNModelInfo model_info;
    model_info.model_file_ = parameters.ModelWeightsPath(i);
    model_info.quantized_ = parameters.IsQuantized(i);

    if (parameters.ModelBufferSize(i) > 0) {
      model_info.model_size_ = parameters.ModelBufferSize(i);
      model_info.model_buffer_ = parameters.ModelWeightsBuffer(i);
      assert(model_info.model_buffer_);

      model_info.model_ =
          shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(
              model_info.model_buffer_, model_info.model_size_));

      /* MNN will save a copy of model buffer. Therefore, release mine */
      RDVLOG(2) << "Destroying model buffer "
                << static_cast<const void*>(model_info.model_buffer_);
      delete[] model_info.model_buffer_;
      model_info.model_buffer_ = nullptr;
      model_info.model_size_ = 0;
    }
    else {
      RDVLOG(1) << "Loading from disk.";
      model_info.model_ = shared_ptr<MNN::Interpreter>(
          MNN::Interpreter::createFromFile(model_info.model_file_.c_str()));

      RDVLOG(1) << "Model loaded. size = "
                << model_info.model_->getModelBuffer().second;
    }

    if (!model_info.model_) {
      return false;
    }

    MNN::ScheduleConfig config;
#ifdef REAL_DEVICE
    //  RDLOG(INFO) << "Specifying OpenCL as computing backend.";
    //  RDLOG(INFO) << "Loading libMNN_CL.so library explicitly.";
    //
    //  auto handle = dlopen("libMNN_CL.so", RTLD_NOW);
    //  FUNC_PRINT_ALL(handle, p);

    if (parameters.RuntimeType() == RUNTIME_GPU) {
      config.type = MNN_FORWARD_OPENCL;
    }
    else if (parameters.RuntimeType() == RUNTIME_CPU) {
      config.type = MNN_FORWARD_CPU;
    }
    else {
      RDLOG(ERROR) << "Invalid MNN runtime type!\n";
      assert(false);
    }
#else
    config.type = MNN_FORWARD_CPU;
#endif

    int num_thread;
#ifdef MNN_THREAD_NUM_1
    num_thread = 1;
#elif MNN_THREAD_NUM_2
    num_thread = 2;
#elif MNN_THREAD_NUM_3
    num_thread = 3;
#elif MNN_THREAD_NUM_4
    num_thread = 4;
#else
#error Invalid MNN thread number!
#endif
    config.numThread = num_thread;
    RDLOG(INFO) << "Using " << num_thread << " threads for inference.";

    MNN::BackendConfig backend_config;
    backend_config.memory =
        MNN::BackendConfig::Memory_Normal;  //Memory_High, Memory_Low
    backend_config.power =
        MNN::BackendConfig::Power_Normal;  //Power_High, Power_Low
    backend_config.precision =
        MNN::BackendConfig::Precision_Normal;  //Precision_High, Precision_Low

    config.backendConfig = &backend_config;

    model_info.session_ = model_info.model_->createSession(config);
    //model and session need to be resized, so the model can't be released, by Wei 230922
    //model_info.model_->releaseModel();

    if (!model_info.session_) {
      return false;
    }

    model_info_list_.emplace_back(model_info);
  }
  return InitializeVocabulary(num_of_models, parameters);
}

bool MNNImpl::InitializeVocabulary(int num_of_models, const BackendParameters& parameters) {
  #ifdef BACKEND_LITE
  #else
  if ((model_type_ == MNN_MODEL_QUARTZNET ||
       model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
       model_type_ == MNN_MODEL_QUARTZNET_REGEX)) {
    if (asr_objs_.language != 0 && asr_objs_.language != 2)
      asr_objs_.vocab_vec.push_back(" ");
    int cfg_model_id;
    if (model_type_ == MNN_MODEL_QUARTZNET_REGEX)
      cfg_model_id = num_of_models + 1;
    else
      cfg_model_id = num_of_models;
    std::string_view mbs((char*)parameters.ModelWeightsBuffer(cfg_model_id),
                    parameters.ModelBufferSize(cfg_model_id));
    size_t pos = 0;
    std::string word;
    std::string delimiter = "\n";
    std::string delim = " ";
    std::string delim_comma = ",";
    int char_id = 0;
    while ((pos = mbs.find(delimiter)) != std::string_view::npos) {
      word = mbs.substr(0, pos);
      mbs.remove_prefix(pos + delimiter.length());
      if (word == "tlg wordlist") {
        break;
      }
      std::vector<std::string> slu_input_info =
          speechengine::quartznet::split(word, delim);
      asr_objs_.vocab_vec.push_back(slu_input_info[0]);
      if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN &&
          (!asr_objs_.use_language_model)) {
        asr_objs_.slu_input.insert(
            std::pair<int, std::string>(char_id, slu_input_info[1]));
        char_id++;
      }

#ifdef USE_LANGUAGE_MODEL
      asr_objs_.num_of_tlgdecs_ =
          ceil(asr_expected_buffer_length[parameters.LanguageType()] /
               asr_expected_chunk_length[parameters.LanguageType()]);
      if ((model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
           model_type_ == MNN_MODEL_QUARTZNET_REGEX) &&
          asr_objs_.use_language_model) {
        asr_objs_.acoustic_language_map.push_back(
            atoi(slu_input_info[2].c_str()));
      }
#endif
    }
    if (asr_objs_.language == 0 || asr_objs_.language == 2) {
      asr_objs_.vocab_vec.push_back(" ");
      asr_objs_.vocab_vec.push_back("''");
    }
    if (asr_objs_.use_language_model) {
#ifdef USE_LANGUAGE_MODEL

      std::string_view model_bytes(parameters.ModelWeightsBuffer(cfg_model_id + 1),
                                   parameters.ModelBufferSize(cfg_model_id + 1));
      auto fstReadOptions = fst::FstReadOptions();
      auto model_bytes_stream = istring_view(model_bytes);
      auto g_decode_resource = std::make_shared<wenet::DecodeResource>();
      auto fst = std::shared_ptr<fst::VectorFst<fst::StdArc>>(
          fst::VectorFst<fst::StdArc>::Read(model_bytes_stream, fstReadOptions));
      assert(fst != nullptr);
      g_decode_resource->fst = fst;
      delete[] parameters.ModelWeightsBuffer(cfg_model_id + 1);

      std::vector<std::string> word_list_vec;
      size_t pos = 0;
      while ((pos = mbs.find(delimiter)) != std::string_view::npos) {
        word = mbs.substr(0, pos);
        std::vector<std::string> slu_input_info =
            speechengine::quartznet::split(word, delim);
        if ((model_type_ == MNN_MODEL_QUARTZNET_CNNATTN &&
             slu_input_info.size() < 3) ||
            slu_input_info.size() < 2) {
          RDVLOG(1) << "the length of slu_input_info is wrong";
          break;
        }
        word_list_vec.push_back(slu_input_info[0] + delim + slu_input_info[1]);
        if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN)
          asr_objs_.slu_input.insert(std::pair<int, std::string>(
              atoi(slu_input_info[1].c_str()), slu_input_info[2]));
        // mbs.erase(0, pos + delimiter.length());
        mbs.remove_prefix(pos + delimiter.length());
      }
      delete[] parameters.ModelWeightsBuffer(cfg_model_id);

      auto symbol_table = std::shared_ptr<fst::SymbolTable>(
          fst::SymbolTable::ReadText(word_list_vec));
      assert(symbol_table != nullptr);
      g_decode_resource->symbol_table = symbol_table;

      std::shared_ptr<wenet::DecodeOptions> g_decode_config =
          InitDecodeOptionsFromFlags();
      if (delay_time[asr_objs_.language] >= 0) {
        std::unique_ptr<wenet::AsrDecoder> asrDec(
            new wenet::AsrDecoder(g_decode_resource, *g_decode_config));
        asr_objs_.decoder_vec_.push_back(std::move(asrDec));
      }
      else {
        for (int kk = 0; kk < asr_objs_.num_of_tlgdecs_; kk++) {
          std::unique_ptr<wenet::AsrDecoder> asrDec(
              new wenet::AsrDecoder(g_decode_resource, *g_decode_config));
          asr_objs_.decoder_vec_.push_back(std::move(asrDec));
        }
      }
#endif
    }
    else {
      delete[] parameters.ModelWeightsBuffer(cfg_model_id);
    }
    if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
      for (int jj = 0;
           jj < preset_regex_patterns[parameters.LanguageType() - 1].size();
           jj++) {
        asr_objs_.regex_patterns.push_back(speechengine::quartznet::split(
            preset_regex_patterns[parameters.LanguageType() - 1][jj],
            delim_comma));
      }
      for (int jj = 0;
           jj < preset_regex_patterns_voiceactive[parameters.LanguageType() - 1]
                    .size();
           jj++) {
        asr_objs_.regex_patterns_voiceactive.push_back(
            speechengine::quartznet::split(
                preset_regex_patterns_voiceactive[parameters.LanguageType() - 1]
                                                 [jj],
                delim_comma));
      }
    }
    speechengine::ASRSLUUtil::reset_asr_status_variables(
        asr_stt_vars_,
        ceil(asr_expected_buffer_length[parameters.LanguageType()] /
             asr_expected_chunk_length[parameters.LanguageType()]));
  }
#ifdef INIT_LOCK
  if (parameters.RuntimeType() == RUNTIME_GPU) {
    mtx.unlock();
  }
#endif
#endif
  initialized_ = true;
  return true;
}


void MNNImpl::Deinitialize()
{
#ifdef MNN_WITH_SPEEX_AEC
  if (model_type_ == MNN_MODEL_SPEEX_AEC) {
    delete aec_objs_.echoCanceller_;
    aec_objs_.echoCanceller_ = nullptr;
  }
#endif

#ifdef MNN_WITH_SIGNAL_DOA
  if (model_type_ == MNN_MODEL_SIGNAL_DOA) {
    delete doa_objs_.doa_obj_;
    doa_objs_.doa_obj_ = nullptr;
  }
#endif
  for (auto model_info : model_info_list_) {
    model_info.model_->releaseModel();
    if (model_info.model_size_ > 0) {
      assert(model_info.model_buffer_);
      RDVLOG(2) << "Destroying model buffer "
                << static_cast<const void*>(model_info.model_buffer_);
      delete[] model_info.model_buffer_;

      model_info.model_size_ = 0;
    }

    if (initialized_) {
      model_info.model_->releaseSession(model_info.session_);
      model_info.session_ = nullptr;

      model_info.model_.reset();
    }
    if (model_type_ == MNN_MODEL_RNNVAD) {
      #ifndef BACKEND_LITE
      vad_objs_.RNNVad_->RNNVadDestroy();
      #endif
      if (vad_objs_.RNNVad_ != nullptr) {
        delete vad_objs_.RNNVad_;
        vad_objs_.RNNVad_ = nullptr;
      }
    }
  }
  if ((model_type_ == MNN_MODEL_QUARTZNET ||
       model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
       model_type_ == MNN_MODEL_QUARTZNET_REGEX)) {
    asr_objs_.vocab_vec.clear();
    if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN) asr_objs_.slu_input.clear();
#ifdef USE_LANGUAGE_MODEL
    if (asr_objs_.use_language_model) {
      #ifndef BACKEND_LITE
      if (delay_time[asr_objs_.language] >= 0) {
        asr_objs_.decoder_vec_[0].reset();
      }
      else {
        for (int kk = 0; kk < asr_objs_.num_of_tlgdecs_; kk++) {
          asr_objs_.decoder_vec_[kk].reset();
        }
      }
      asr_objs_.decoder_vec_.clear();
      #endif
      asr_objs_.acoustic_language_map.clear();
    }
#endif
    if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
      asr_objs_.regex_patterns.clear();
      asr_objs_.regex_patterns_voiceactive.clear();
    }
  }
  initialized_ = false;
}


void MNNImpl::Invoke(const float* audio_data, int sample_rate,
                     int buffer_length, int chunk_length, int channels)
{
// model_type_ = MNN_MODEL_WUW_RESNET;
#ifdef MNN_WAKEWORD_MODEL_PRECISE
  if (model_type_ == MNN_MODEL_PRECISE) {
    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    precise::model_invoke(model, session, audio_data, sample_rate,
                          buffer_length);
  }
#elif defined MNN_WAKEWORD_MODEL_RESNET
  if (model_type_ == MNN_MODEL_WUW_RESNET) {
    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    speechengine::rrwuw::model_invoke(model, session, audio_data, sample_rate,
                                      buffer_length);
  }
#else
  if (model_type_ == MNN_MODEL_WUW_RESNET) {
    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    speechengine::rrwuw::model_invoke(model, session, audio_data, sample_rate,
                                      buffer_length);
  }
#endif
  else if (model_type_ == MNN_MODEL_QUARTZNET ||
           model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
           model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
    #ifndef BACKEND_LITE
    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;
    quartznet::model_invoke(model, session, audio_data, sample_rate,
                            buffer_length, chunk_length);
    #endif
  }

}


/* Wakeword evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       WakewordResult* o_Results,
                       std::function<void(WakewordResult*)> cb)
{
  TensorDict outputs;

  if (model_type_ == MNN_MODEL_PRECISE) {
    // precise wakeword model only have 1 model file
    assert(model_info_list_.size() == 1);

    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    const string output_tensor_name = "score_predict/Sigmoid";
    outputs[output_tensor_name] =
        model->getSessionOutput(session, output_tensor_name.c_str());
  }
  else if (model_type_ == MNN_MODEL_WUW_RESNET) {
    // precise wakeword model only have 1 model file
    assert(model_info_list_.size() == 1);

    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    // const string output_tensor_name = "191";
    const string output_tensor_name = "output";
    outputs[output_tensor_name] =
        model->getSessionOutput(session, output_tensor_name.c_str());
    // auto outputTensor = model->getSessionOutput (session, NULL);
  }

  Invoke(audio_data, sample_rate, buffer_length, chunk_length, channels);

  speechengine::rrwuw::GetWakewordResults(outputs, o_Results, chunk_length);

  if (o_Results->activate) cb(o_Results);
  return;
}


/* Denoise evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       DenoiseResult* o_Results)
{
  TensorDict outputs;

#if 0
  if (model_type_ == MNN_MODEL_PRECISE) {
    // precise wakeword model only have 1 model file
    assert (model_info_list_.size() == 1);

    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    const string output_tensor_name = "score_predict/Sigmoid";
    outputs[output_tensor_name] = model->getSessionOutput (session, output_tensor_name.c_str());
  }

  Invoke (audio_data, sample_rate, buffer_length, chunk_length, channels);

  GetWakewordResults(
          outputs, o_Results);
#endif

  return;
}


/* DOA evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       bool is_last_frame, DOAResult* o_Results)
{
#ifndef BACKEND_LITE
#ifdef USE_SIGNAL_DOA
  if (model_type_ == MNN_MODEL_SIGNAL_DOA) {
    float doa_peaks[4] = {-1.0f, -1.0f, 0.0f, 0.0f};
    rrse_SignalDOA::DOA_Matrix* doa_results = new rrse_SignalDOA::DOA_Matrix();
    doa_objs_.doa_obj_->process(audio_data, channels, chunk_length,
                                is_last_frame, doa_results);
    o_Results->angle = doa_results->doa1;
    o_Results->sec_angle = doa_results->doa2;
    o_Results->confidence = doa_results->doa1_energy_percentage;
    delete doa_results;
    // o_Results.
    // float angle;
    // float sec_angle;
    // float confidence;
  }
#endif
#endif
  return;
}


/* AEC evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       AECResult* o_Results)
{
  TensorDict outputs;

#ifdef MNN_WITH_SPEEX_AEC
  if (model_type_ == MNN_MODEL_SPEEX_AEC) {
    float** output_buffer_ptr = &(o_Results->output_buffer);
    float** output_echo_buffer_ptr = &(o_Results->output_echo_buffer);
    aec_objs_.echoCanceller_->process(audio_data, output_buffer_ptr, channels,
                                      chunk_length);
    aec_objs_.echoCanceller_->getEchoFloatFrame(output_echo_buffer_ptr);
  }
#if USE_AEC_NN_MODEL
  if (model_type_ == MNN_MODEL_PRECISE) {
    // precise wakeword model only have 1 model file
    assert(model_info_list_.size() == 1);

    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    const string output_tensor_name = "score_predict/Sigmoid";
    outputs[output_tensor_name] =
        model->getSessionOutput(session, output_tensor_name.c_str());
  }

  Invoke(audio_data, sample_rate, buffer_length, chunk_length, channels);

  GetWakewordResults(outputs, o_Results);
#endif
#endif

  return;
}

#ifndef BACKEND_LITE
/* VAD evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       VADResult* o_Results)
{
#ifndef BACKEND_LITE
  if (model_type_ == MNN_MODEL_RNNVAD) {
    int rnnvad_framesize = RNNVAD_FRAME_SIZE;
    float* x = new float[rnnvad_framesize];
    for (int i = 0; i < rnnvad_framesize; i++) {
      x[i] = audio_data[i] * 32768;
    }
    int onlyvad_running = 1;
    float vad_prob = 0.f;
    RNNVAD_RESULTS res = vad_objs_.RNNVad_->VadProcessFrame(x, vad_prob);
    o_Results->activate = res.speech_trigger;
    o_Results->confidence = vad_prob;
    delete[] x;
  }
  else {
    std::cerr << "FILES: " << __FILE__ << " LINES: " << __LINE__ << std::endl;
    std::cerr << "ERROR: ONLY RNNVAD support." << std::endl;
  }
#endif
  return;
}

/* ASR evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       int is_validate, ASRResult* o_Results,
                       std::function<void(ASRResult*)> cb)
{
  static std::vector<int> asr_output;
  static bool asr_validated = false;
  o_Results->is_ready = false;
  if (is_validate == 0 && !asr_validated) return;

  if (!asr_validated) {
    if (is_validate > 0) {
#ifdef USE_LANGUAGE_MODEL
      if (model_type_ == MNN_MODEL_QUARTZNET && asr_objs_.use_language_model) {
        asr_objs_.decoder_vec_[0]->Reset();
      }
#endif
    }
    else {
      RDLOG(ERROR) << "run into a state that is not expected\n";
      return;
    }
  }
  if (!(is_validate == 0 && asr_validated)) {
    TensorDict outputs;
    if (model_type_ == MNN_MODEL_QUARTZNET) {
      // quartznet asr model only have 1 acoustic model file
      // assert (model_info_list_.size() == 1);

      std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
      MNN::Session* session = model_info_list_[0].session_;

      const string output_tensor_name = "logprobs";
      outputs[output_tensor_name] =
          model->getSessionOutput(session, output_tensor_name.c_str());
    }
    Invoke(audio_data, sample_rate, buffer_length, chunk_length, channels);
    if (model_type_ == MNN_MODEL_QUARTZNET) {
      if (asr_objs_.use_language_model) {
#ifdef USE_LANGUAGE_MODEL
        std::vector<std::vector<float>> probs_seq =
            speechengine::quartznet::GetASRResults(
                outputs, buffer_length, chunk_length,
                asr_objs_.vocab_vec.size() + 1, asr_objs_.acoustic_language_map,
                asr_objs_.language);
        asr_objs_.decoder_vec_[0]->Decode(probs_seq);
#endif
      }
      else {
        std::vector<std::vector<float>> probs_seq =
            speechengine::quartznet::GetASRResults(
                outputs, buffer_length, chunk_length,
                asr_objs_.vocab_vec.size() + 1, asr_objs_.language);
        std::vector<int> decode_result =
            speechengine::quartznet::_greedy_decoder(
                probs_seq, asr_objs_.vocab_vec.size());
        asr_output.insert(asr_output.end(), decode_result.begin(),
                          decode_result.end());
      }
    }
  }
  if (is_validate < 1 && asr_validated) {
    if (model_type_ == MNN_MODEL_QUARTZNET) {
      if (asr_objs_.use_language_model) {
#ifdef USE_LANGUAGE_MODEL
        // asr_objs_.decoder_->Rescoring();
        // std::cout<<"Score:"<<bs_result[0].first<<"\n";
        // std::cout<<"Script:"<<bs_result[0].second<<"\n";
        if (asr_objs_.decoder_vec_[0]->DecodedSomething()) {
          // std::cout<<"decoder result:"<<asr_objs_.decoder_->result()[0].sentence<<"\n";
          int res_len =
              asr_objs_.decoder_vec_[0]->result()[0].sentence.length();
          asr_objs_.decoder_vec_[0]->result()[0].sentence.copy(
              o_Results->output_string, res_len, 0);
          *(o_Results->output_string + res_len) = '\0';
        }
        else {
          *(o_Results->output_string) = '\0';
        }
#endif
      }
      else {
        speechengine::quartznet::greedy_merge(asr_output, asr_objs_.vocab_vec,
                                              o_Results->output_string);
        asr_output.clear();
      }
    }

    o_Results->is_ready = true;
    cb(o_Results);
  }
  asr_validated = is_validate > 0;
  return;
}

/* ASR with SLU evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       int is_validate, ASRSLUResult* o_Results,
                       std::function<void(ASRSLUResult*)> cb)
{
  int intent_res = 1, tmpintent = 1,
      buf_stride = ceil(chunk_length / (int(sample_rate * 0.01 + 0.5) * 2)),
      words_needed = 0, past_inuse_size = 0,
      asr_buffer_num = ceil(buffer_length / chunk_length),
      buffer_stride = ceil(buffer_length / (int(sample_rate * 0.01 + 0.5) * 2));
  std::string delim = ",", delim_blank = " ", tmp_output_string = "",
              past_output_sentences_tmp = "", buffer_output = "";
  bool is_past_full = false;
  float tmpscore = 0.f, intent_value = 0.f;
  std::vector<int> slots, source_ids_withpastinuse, source_ids_reverse,
      source_ids_tlg;
  std::vector<std::vector<int>> source_word_ids, source_ids_past_inuse;
  o_Results->is_ready = false;
  if (is_validate == 0 && !asr_stt_vars_.asr_validated) {
    return;
  }
  if (!asr_stt_vars_.asr_validated) {
    asr_stt_vars_.probs_seq_past.clear();
#ifdef USE_LANGUAGE_MODEL
    if ((model_type_ == MNN_MODEL_QUARTZNET ||
         model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
         model_type_ == MNN_MODEL_QUARTZNET_REGEX) &&
        asr_objs_.use_language_model) {
      if (delay_time[asr_objs_.language] < 0) {
        for (int kk = 0; kk < asr_objs_.num_of_tlgdecs_; kk++) {
          asr_objs_.decoder_vec_[kk]->Reset();
        }
      }
      else
        asr_objs_.decoder_vec_[0]->Reset();
    }
#endif
  }
  if (!(is_validate == 0 && asr_stt_vars_.asr_validated)) {
    TensorDict outputs;
    if (model_type_ == MNN_MODEL_QUARTZNET ||
        model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
        model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
      outputs["logprobs"] = model_info_list_[0].model_->getSessionOutput(
          model_info_list_[0].session_, "logprobs");
    }
    Invoke(audio_data, sample_rate, buffer_length, chunk_length, channels);
    if (model_type_ == MNN_MODEL_QUARTZNET ||
        model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
        model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
      if (asr_objs_.use_language_model) {
#ifdef USE_LANGUAGE_MODEL
        if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN) {
          std::vector<std::vector<float>> probs_seq =
              speechengine::quartznet::GetASRResults(
                  outputs, buffer_length, chunk_length,
                  asr_objs_.vocab_vec.size() + 1,
                  asr_objs_.acoustic_language_map, asr_objs_.language);
          asr_stt_vars_.probs_seq_past.insert(
              asr_stt_vars_.probs_seq_past.end(), probs_seq.begin(),
              probs_seq.end());
          if (asr_stt_vars_.probs_seq_past.size() / buf_stride >=
              buffer_overflow_length) {
            asr_stt_vars_.probs_seq_past.erase(
                asr_stt_vars_.probs_seq_past.begin(),
                asr_stt_vars_.probs_seq_past.begin() +
                    (buffer_overflow_length - 4) * buf_stride);
            asr_objs_.decoder_vec_[0]->Reset();
            asr_objs_.decoder_vec_[0]->Decode(asr_stt_vars_.probs_seq_past);
            //not enough time to deal allpast in lm-decoding-> put even earlier lm-decoded result together with present result into nlu
            for (int jj = asr_stt_vars_.source_ids_past_lm.size() - 1; jj >= 0;
                 jj--) {
              source_ids_past_inuse.push_back(
                  asr_stt_vars_.source_ids_past_lm[jj]);
              past_inuse_size += asr_stt_vars_.source_ids_past_lm[jj].size();
              if (past_inuse_size >= 8) break;
            }
            if (asr_objs_.decoder_vec_[0]->DecodedSomething() &&
                past_inuse_size > 0) {
              speechengine::CNNAttention::AssembleSourceIds(
                  asr_objs_.slu_input,
                  asr_objs_.decoder_vec_[0]->result()[0].hypotheses,
                  source_ids_withpastinuse, delim);
              for (int jj = 0; jj < source_ids_past_inuse.size(); jj++) {
                source_ids_withpastinuse.insert(
                    source_ids_withpastinuse.begin(),
                    source_ids_past_inuse[jj].begin(),
                    source_ids_past_inuse[jj].end());
                intent_res = speechengine::CNNAttention::IntentSlotInfer(
                    model_info_list_[1], source_ids_withpastinuse, intent_value,
                    slots);
                if (intent_res > 1 && intent_value > intent_score_thres_high) {
		  std::cout << "IntentSlotInfer input: ";
		  for (auto source_id : source_ids_withpastinuse) {
		    std::cout << source_id << " ";
		  }
		  std::cout << std::endl;
                  std::vector<std::string> output_words_past =
                      speechengine::quartznet::split(
                          asr_stt_vars_.output_sentence_past, delim_blank);
                  std::string output_sentence =
                      asr_objs_.decoder_vec_[0]->result()[0].sentence;
                  for (int kk = 0; kk <= jj; kk++) {
                    output_sentence =
                        output_words_past[output_words_past.size() - 1 - kk] +
                        output_sentence;
                  }
                  speechengine::CNNAttention::AssembleOutputResults(
                      output_sentence, intent_res, slots, o_Results);
                  asr_objs_.decoder_vec_[0]->Reset();
                  speechengine::quartznet::reset_status(
                      true, o_Results, asr_stt_vars_.probs_seq_past, cb);
                  break;
                }
              }
            }
            asr_stt_vars_.source_ids_past_lm.clear();
            asr_stt_vars_.output_sentence_past = "";
          }
          else {
            asr_objs_.decoder_vec_[0]->Decode(probs_seq);
            if (asr_stt_vars_.probs_seq_past.size() / buf_stride == 11) {
              if (asr_objs_.decoder_vec_[0]->DecodedSomething()) {
                asr_stt_vars_.source_ids_past_lm.clear();
                int total_characs =
                    speechengine::CNNAttention::AssembleSourceIds(
                        asr_objs_.slu_input,
                        asr_objs_.decoder_vec_[0]->result()[0].hypotheses,
                        asr_stt_vars_.source_ids_past_lm, delim, words_needed,
                        false);
                asr_stt_vars_.output_sentence_past =
                    asr_objs_.decoder_vec_[0]->result()[0].sentence;
              }
            }
          }
        }
        else if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
          if (delay_time[asr_objs_.language] < 0) {
            std::vector<std::vector<float>> probs_seq =
                speechengine::quartznet::GetASRResults(
                    outputs, buffer_length, asr_objs_.vocab_vec.size() + 1,
                    asr_objs_.acoustic_language_map, asr_objs_.language);
            if (asr_stt_vars_.num_of_tlgdecode[asr_stt_vars_.asr_buffer_id] >=
                5) {
              if (asr_objs_.decoder_vec_[asr_stt_vars_.asr_buffer_id]
                      ->DecodedSomething()) {
                std::vector<std::string> words_in_sentence =
                    speechengine::quartznet::split(
                        asr_objs_.decoder_vec_[asr_stt_vars_.asr_buffer_id]
                            ->result()[0]
                            .sentence,
                        delim_blank);
                for (int jj = words_in_sentence.size() - 1; jj >= 0; jj--) {
                  past_output_sentences_tmp.insert(
                      past_output_sentences_tmp.begin(),
                      words_in_sentence[jj].begin(),
                      words_in_sentence[jj].end());
                  past_inuse_size += words_in_sentence[jj].length();
                  if (past_inuse_size >= 8) break;
                }
                std::cout << past_output_sentences_tmp
                          << " is past_output_sentences_tmp\n";
              }
              asr_objs_.decoder_vec_[asr_stt_vars_.asr_buffer_id]->Reset();
              asr_stt_vars_.num_of_tlgdecode[asr_stt_vars_.asr_buffer_id] = 0;
            }
            asr_objs_.decoder_vec_[asr_stt_vars_.asr_buffer_id]->Decode(
                probs_seq);
            asr_stt_vars_.num_of_tlgdecode[asr_stt_vars_.asr_buffer_id]++;
            asr_stt_vars_.asr_buffer_id =
                (asr_stt_vars_.asr_buffer_id >= asr_objs_.num_of_tlgdecs_ - 1)
                    ? 0
                    : (asr_stt_vars_.asr_buffer_id + 1);
          }
          else {
            std::vector<std::vector<float>> probs_seq =
                speechengine::quartznet::GetASRResults(
                    outputs, buffer_length, chunk_length,
                    asr_objs_.vocab_vec.size() + 1,
                    asr_objs_.acoustic_language_map, asr_objs_.language);
            asr_objs_.decoder_vec_[0]->Decode(probs_seq);
          }
        }
#endif
      }
      else {
        if (delay_time[asr_objs_.language] < 0) {
          std::vector<std::vector<float>> probs_seq =
              speechengine::quartznet::GetASRResults(
                  outputs, buffer_length, asr_objs_.vocab_vec.size() + 1,
                  asr_objs_.language);
          std::vector<int> decode_result =
              speechengine::quartznet::_greedy_decoder(
                  probs_seq, asr_objs_.vocab_vec.size());
          if (asr_stt_vars_.asr_buffer_outputs[asr_stt_vars_.asr_buffer_id]
                  .size() >= 5 * buffer_stride) {
            speechengine::quartznet::greedy_merge(
                asr_stt_vars_.asr_buffer_outputs[asr_stt_vars_.asr_buffer_id],
                asr_objs_.vocab_vec, buffer_output);
            past_output_sentences_tmp =
                speechengine::RegexMatch::getLastNCharacters(buffer_output, 8);
            asr_stt_vars_.asr_buffer_outputs[asr_stt_vars_.asr_buffer_id]
                .clear();
          }
          asr_stt_vars_.asr_buffer_outputs[asr_stt_vars_.asr_buffer_id].insert(
              asr_stt_vars_.asr_buffer_outputs[asr_stt_vars_.asr_buffer_id]
                  .end(),
              decode_result.begin(), decode_result.end());
          asr_stt_vars_.asr_buffer_id =
              (asr_stt_vars_.asr_buffer_id >= asr_buffer_num - 1)
                  ? 0
                  : (asr_stt_vars_.asr_buffer_id + 1);
        }
        else {
          std::vector<std::vector<float>> probs_seq =
              speechengine::quartznet::GetASRResults(
                  outputs, buffer_length, chunk_length,
                  asr_objs_.vocab_vec.size() + 1, asr_objs_.language);
          asr_stt_vars_.probs_seq_past.insert(
              asr_stt_vars_.probs_seq_past.end(), probs_seq.begin(),
              probs_seq.end());
          is_past_full = asr_stt_vars_.probs_seq_past.size() / buf_stride >=
                         buffer_overflow_length_nolm;
          if (is_past_full) {
            asr_stt_vars_.probs_seq_past.erase(
                asr_stt_vars_.probs_seq_past.begin(),
                asr_stt_vars_.probs_seq_past.begin() +
                    (buffer_overflow_length_nolm - 8) * buf_stride);
            std::vector<int> greedy_decoder_res =
                speechengine::quartznet::_greedy_decoder(
                    asr_stt_vars_.probs_seq_past, asr_objs_.vocab_vec.size());
            asr_stt_vars_.asr_output.clear();
            asr_stt_vars_.asr_output.swap(greedy_decoder_res);
          }
          else {
            std::vector<int> decode_result =
                speechengine::quartznet::_greedy_decoder(
                    probs_seq, asr_objs_.vocab_vec.size());
            asr_stt_vars_.asr_output.insert(asr_stt_vars_.asr_output.end(),
                                            decode_result.begin(),
                                            decode_result.end());
          }
        }
      }
    }
  }
  if (is_validate == -1 || (is_validate == 0 && asr_stt_vars_.asr_validated)) {
    if (model_type_ == MNN_MODEL_QUARTZNET ||
        model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
        model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
      if (asr_objs_.use_language_model) {
#ifdef USE_LANGUAGE_MODEL
        *(o_Results->output_string) = '\0';
        o_Results->intent = 1;
        for (int ii = 0; ii < intentslot_source_length; ii++)
          o_Results->slots[ii] = 4;
        if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN &&
            asr_objs_.decoder_vec_[0]->DecodedSomething()) {
          if(multi_cnnattcall_on_vaddrop){
            asr_objs_.decoder_vec_[0]->result()[0].sentence.copy(
                o_Results->output_string,
                asr_objs_.decoder_vec_[0]->result()[0].sentence.length(), 0);
            *(o_Results->output_string +
              asr_objs_.decoder_vec_[0]->result()[0].sentence.length()) = '\0';
            words_needed = 0;
            int total_characs = speechengine::CNNAttention::AssembleSourceIds(
                asr_objs_.slu_input,
                asr_objs_.decoder_vec_[0]->result()[0].hypotheses,
                source_word_ids, delim, words_needed, true);
	    std::cout << "ASRSLU buffer length = " << buffer_length << " audio_data = " << audio_data << std::endl;
            for (int ii = 0; ii < source_word_ids.size(); ii++) {
              source_ids_reverse.insert(source_ids_reverse.begin(),
                                        source_word_ids[ii].begin(),
                                        source_word_ids[ii].end());
              intent_res = speechengine::CNNAttention::IntentSlotInfer(
                  model_info_list_[1], source_ids_reverse, intent_value, slots);
	      std::cout << __FILE__ << ":" << __LINE__ << " ii = " << ii << " source_ids:";
	      std::copy(source_ids_reverse.begin(), source_ids_reverse.end(),
			std::ostream_iterator<int>(std::cout, ", "));
	      std::cout << std::endl;
              speechengine::CNNAttention::UpdateStatusBySemanticRichness(
                  intent_res, intent_value, o_Results, slots, tmpintent,
                  tmpscore);
            }
            std::cout << asr_objs_.decoder_vec_[0]->result()[0].sentence
                      << " is asr result by tlg " << __LINE__ << "\n";
            if (tmpintent > 1) {
              if (tmpscore > 12.5f) {
                std::cout << "effective intent obtained\n";
              }
              else {
                std::cout << tmpscore
                          << " effective intent not obtained because of score\n";
                o_Results->intent = 1;
              }
            }
          }
          else{
            speechengine::CNNAttention::AssembleSourceIds(
              asr_objs_.slu_input,
              asr_objs_.decoder_vec_[0]->result()[0].hypotheses, source_ids_tlg,
              delim);
            intent_res = speechengine::CNNAttention::IntentSlotInfer(
                  model_info_list_[1], source_ids_tlg, intent_value,
                  slots);
            std::cout << asr_objs_.decoder_vec_[0]->result()[0].sentence
		      << " is asr result by tlg" << __LINE__ << "\n";
            speechengine::CNNAttention::AssembleOutputResults(
              asr_objs_.decoder_vec_[0]->result()[0].sentence, intent_res,
              slots, o_Results);
          }
        }
        else if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
          if (delay_time[asr_objs_.language] < 0) {
            bool tlgres_decoded = false;
            for (int kk = 0; kk < asr_objs_.num_of_tlgdecs_; kk++) {
              if (asr_objs_.decoder_vec_[kk]->DecodedSomething()) {
                tlgres_decoded = true;
                break;
              }
            }
            if (tlgres_decoded) {
              for (int kk = 0; kk < asr_objs_.num_of_tlgdecs_; kk++) {
                if (!asr_objs_.decoder_vec_[kk]->DecodedSomething()) continue;
                std::string decoded_str =
                    asr_objs_.decoder_vec_[kk]->result()[0].sentence;
                decoded_str.erase(
                    std::remove(decoded_str.begin(), decoded_str.end(), ' '),
                    decoded_str.end());
                intent_res = speechengine::RegexMatch::regex_match(
                    decoded_str, asr_objs_.regex_patterns);
                if (intent_res > 1) {
                  if (intent_semantic_richness[intent_res] >
                          intent_semantic_richness[tmpintent] ||
                      (intent_semantic_richness[intent_res] ==
                           intent_semantic_richness[tmpintent] &&
                       decoded_str.length() > tmp_output_string.length())) {
                    tmpintent = intent_res;
                    tmp_output_string = decoded_str;
                  }
                }
              }
              std::cout << tmp_output_string << " is asr result";
              speechengine::CNNAttention::AssembleOutputResults(
                  tmp_output_string, tmpintent, slots, o_Results, false);
            }
          }
          else {
            if (asr_objs_.decoder_vec_[0]->DecodedSomething()) {
              bool tmp_res = speechengine::CNNAttention::regex_match(
                  asr_objs_.decoder_vec_[0]->result()[0].sentence,
                  asr_objs_.regex_patterns,
                  allowed_max_spans[asr_objs_.language - 1], asr_objs_.language,
                  false, o_Results);
            }
          }
        }
        if (model_type_ == MNN_MODEL_QUARTZNET_REGEX &&
            delay_time[asr_objs_.language] < 0) {
          asr_stt_vars_.asr_buffer_id = 0;
          for (int ii = 0; ii < asr_objs_.num_of_tlgdecs_; ii++)
            asr_stt_vars_.num_of_tlgdecode[ii] = 0;
        }
#endif
      }
      else {
        if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN) {
          std::vector<int> source_ids =
              speechengine::quartznet::greedy_merge_with_sourceid(
                  asr_stt_vars_.asr_output, asr_objs_.vocab_vec,
                  asr_objs_.slu_input, o_Results->output_string);
          o_Results->intent = speechengine::CNNAttention::IntentSlotInfer(
              model_info_list_[1], source_ids, intent_value, slots);
          for (int ii = 0; ii < intentslot_source_length; ii++)
            o_Results->slots[ii] = slots[ii];
        }
        else if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
          if (delay_time[asr_objs_.language] < 0) {
            for (int kk = 0; kk < asr_buffer_num; kk++) {
              speechengine::quartznet::greedy_merge(
                  asr_stt_vars_.asr_buffer_outputs[kk], asr_objs_.vocab_vec,
                  buffer_output);
              intent_res = speechengine::RegexMatch::regex_match(
                  buffer_output, asr_objs_.regex_patterns);
              if (intent_res > 1) {
                if (intent_semantic_richness[intent_res] >
                        intent_semantic_richness[tmpintent] ||
                    (intent_semantic_richness[intent_res] ==
                         intent_semantic_richness[tmpintent] &&
                     buffer_output.length() > tmp_output_string.length())) {
                  tmpintent = intent_res;
                  tmp_output_string = buffer_output;
                }
              }
            }
            speechengine::CNNAttention::AssembleOutputResults(
                tmp_output_string, tmpintent, slots, o_Results, false);
          }
          else {
            speechengine::quartznet::greedy_merge(
                asr_stt_vars_.asr_output, asr_objs_.vocab_vec, buffer_output);
            std::cout << buffer_output << " is buffer_output\n";
            bool tmp_res = speechengine::CNNAttention::regex_match(
                buffer_output, asr_objs_.regex_patterns,
                allowed_max_spans[asr_objs_.language - 1], asr_objs_.language,
                false, o_Results);
          }
        }
        if (model_type_ == MNN_MODEL_QUARTZNET_REGEX &&
            delay_time[asr_objs_.language] < 0) {
          asr_stt_vars_.asr_buffer_id = 0;
          asr_stt_vars_.asr_buffer_outputs.clear();
          asr_stt_vars_.asr_buffer_outputs.resize(asr_buffer_num);
        }
        else
          asr_stt_vars_.asr_output.clear();
      }
    }
    speechengine::quartznet::reset_status(false, o_Results,
                                          asr_stt_vars_.probs_seq_past, cb);
  }
  else {
#ifdef NLU_VOICE_ACTIVE
    if (asr_objs_.use_language_model) {
#ifdef USE_LANGUAGE_MODEL
      if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN) {
        if (asr_objs_.decoder_vec_[0]->DecodedSomething()) {
          asr_objs_.decoder_vec_[0]->result()[0].sentence.copy(
              o_Results->output_string,
              asr_objs_.decoder_vec_[0]->result()[0].sentence.length(), 0);
          *(o_Results->output_string +
            asr_objs_.decoder_vec_[0]->result()[0].sentence.length()) = '\0';
          words_needed = 0;
          int total_characs = speechengine::CNNAttention::AssembleSourceIds(
              asr_objs_.slu_input,
              asr_objs_.decoder_vec_[0]->result()[0].hypotheses,
              source_word_ids, delim, words_needed, true);
          if (source_word_ids != asr_stt_vars_.source_ids_past_words &&
              (total_characs >= 8 ||
               asr_stt_vars_.probs_seq_past.size() / buf_stride >=
                   (buffer_overflow_length - 1))) {
            asr_stt_vars_.source_ids_past_words.clear();
            for (int ii = 0; ii < words_needed - 1; ii++) {
              source_ids_reverse.insert(source_ids_reverse.begin(),
                                        source_word_ids[ii].begin(),
                                        source_word_ids[ii].end());
            }
            for (int ii = words_needed - 1; ii < source_word_ids.size(); ii++) {
              source_ids_reverse.insert(source_ids_reverse.begin(),
                                        source_word_ids[ii].begin(),
                                        source_word_ids[ii].end());
              intent_res = speechengine::CNNAttention::IntentSlotInfer(
                  model_info_list_[1], source_ids_reverse, intent_value, slots);
              speechengine::CNNAttention::UpdateStatusBySemanticRichness(
                  intent_res, intent_value, o_Results, slots, tmpintent,
                  tmpscore);
            }
            if (tmpintent > 1 && tmpscore > intent_score_thres_low) {
              std::cout << asr_objs_.decoder_vec_[0]->result()[0].sentence
                        << " is asr result by tlg in nva " << __LINE__ << "\n";
              asr_objs_.decoder_vec_[0]->Reset();
              speechengine::quartznet::reset_status(
                  true, o_Results, asr_stt_vars_.probs_seq_past, cb);
            }
            else {
              asr_stt_vars_.source_ids_past_words.insert(
                  asr_stt_vars_.source_ids_past_words.end(),
                  source_word_ids.begin(), source_word_ids.end());
            }
          }
        }
      }
      else if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
        if (delay_time[asr_objs_.language] < 0) {
          int recent_buf_id = (asr_stt_vars_.asr_buffer_id - 1 >= 0)
                                  ? asr_stt_vars_.asr_buffer_id - 1
                                  : asr_objs_.num_of_tlgdecs_ - 1;
          if (asr_objs_.decoder_vec_[recent_buf_id]->DecodedSomething()) {
            std::string decoded_str =
                asr_objs_.decoder_vec_[recent_buf_id]->result()[0].sentence;
            decoded_str.erase(
                std::remove(decoded_str.begin(), decoded_str.end(), ' '),
                decoded_str.end());
            if (asr_stt_vars_.num_of_tlgdecode[recent_buf_id] <= 1 &&
                !past_output_sentences_tmp.empty())
              decoded_str.insert(decoded_str.begin(),
                                 past_output_sentences_tmp.begin(),
                                 past_output_sentences_tmp.end());
            std::cout << decoded_str << " is decoded_str\n";
            if (speechengine::RegexMatch::str_len(decoded_str) >= 13) {
              intent_res = speechengine::CNNAttention::regex_match(
                  decoded_str, asr_objs_.regex_patterns_voiceactive, true,
                  o_Results);
              if (intent_res > 1) {
                speechengine::quartznet::reset_status(
                    false, o_Results, asr_stt_vars_.probs_seq_past, cb);
                asr_stt_vars_.asr_buffer_id = 0;
                for (int ii = 0; ii < asr_objs_.num_of_tlgdecs_; ii++) {
                  asr_objs_.decoder_vec_[ii]->Reset();
                  asr_stt_vars_.num_of_tlgdecode[ii] = 0;
                }
              }
            }
          }
        }
      }
#endif
    }
    else {
      if (model_type_ == MNN_MODEL_QUARTZNET_CNNATTN) {
        std::vector<int> source_ids =
            speechengine::quartznet::greedy_merge_with_sourceid(
                asr_stt_vars_.asr_output, asr_objs_.vocab_vec,
                asr_objs_.slu_input, o_Results->output_string);
        if ((source_ids.size() >= 8 || is_past_full) &&
            source_ids != asr_stt_vars_.source_ids_past) {
          asr_stt_vars_.source_ids_past.clear();
          for (int ii = 3; ii <= source_ids.size(); ii++) {
            std::vector<int> source_ids_tmp(source_ids.end() - ii,
                                            source_ids.end());
            intent_res = speechengine::CNNAttention::IntentSlotInfer(
                model_info_list_[1], source_ids_tmp, intent_value, slots);
            speechengine::CNNAttention::UpdateStatusBySemanticRichness(
                intent_res, intent_value, o_Results, slots, tmpintent,
                tmpscore);
          }
          if (tmpintent > 1 && tmpscore > intent_score_thres_high) {
            asr_stt_vars_.asr_output.clear();
            speechengine::quartznet::reset_status(
                true, o_Results, asr_stt_vars_.probs_seq_past, cb);
          }
          else {
            asr_stt_vars_.source_ids_past.insert(
                asr_stt_vars_.source_ids_past.end(), source_ids.begin(),
                source_ids.end());
          }
        }
      }
      else if (model_type_ == MNN_MODEL_QUARTZNET_REGEX) {
        if (delay_time[asr_objs_.language] < 0) {
          int recent_buf_id = (asr_stt_vars_.asr_buffer_id - 1 >= 0)
                                  ? asr_stt_vars_.asr_buffer_id - 1
                                  : asr_buffer_num - 1;
          speechengine::quartznet::greedy_merge(
              asr_stt_vars_.asr_buffer_outputs[recent_buf_id],
              asr_objs_.vocab_vec, buffer_output);
          if (asr_stt_vars_.asr_buffer_outputs[recent_buf_id].size() <=
                  buffer_stride &&
              !past_output_sentences_tmp.empty())
            buffer_output.insert(buffer_output.begin(),
                                 past_output_sentences_tmp.begin(),
                                 past_output_sentences_tmp.end());
          if (speechengine::RegexMatch::str_len(buffer_output) >= 13) {
            intent_res = speechengine::CNNAttention::regex_match(
                buffer_output, asr_objs_.regex_patterns_voiceactive, true,
                o_Results);
            if (intent_res > 1) {
              speechengine::quartznet::reset_status(
                  false, o_Results, asr_stt_vars_.probs_seq_past, cb);
              asr_stt_vars_.asr_buffer_id = 0;
              asr_stt_vars_.asr_buffer_outputs.clear();
              asr_stt_vars_.asr_buffer_outputs.resize(asr_buffer_num);
            }
          }
        }
        else {
          speechengine::quartznet::greedy_merge(
              asr_stt_vars_.asr_output, asr_objs_.vocab_vec, buffer_output);
          int overflow_length = asr_objs_.language == 2 ? 39 : 35;
          if (is_past_full || buffer_output.length() > overflow_length) {
            intent_res = speechengine::CNNAttention::regex_match(
                buffer_output, asr_objs_.regex_patterns_voiceactive,
                allowed_max_spans[asr_objs_.language - 1], asr_objs_.language,
                true, o_Results);
            if (intent_res > 1) {
              asr_stt_vars_.asr_output.clear();
              speechengine::quartznet::reset_status(
                  true, o_Results, asr_stt_vars_.probs_seq_past, cb);
            }
          }
        }
      }
    }
#endif
  }
  asr_stt_vars_.asr_validated = is_validate > 0;
  return;
}

/* ASR with SLU state varible reseting */
void MNNImpl::ResetASRSLUStateVars(int buffer_length, int chunk_length)
{
  speechengine::ASRSLUUtil::reset_asr_status_variables(
      asr_stt_vars_, ceil(buffer_length / chunk_length));
}

/* KWS evaluate */
void MNNImpl::Evaluate(const float* audio_data, int sample_rate,
                       int buffer_length, int chunk_length, int channels,
                       KWSResult* o_Results, std::function<void(KWSResult*)> cb)
{
  TensorDict outputs;

#if 0
  if (model_type_ == MNN_MODEL_PRECISE_KWS) {
    // precise kws model only have 1 model file
    assert (model_info_list_.size() == 1);

    std::shared_ptr<MNN::Interpreter> model = model_info_list_[0].model_;
    MNN::Session* session = model_info_list_[0].session_;

    const string output_tensor_name = "score_predict/Softmax";
    outputs[output_tensor_name] = model->getSessionOutput (session, output_tensor_name.c_str());
  }

  Invoke (audio_data, sample_rate, buffer_length, chunk_length, channels);

  GetKWSResults(
          outputs, o_Results);
#endif
  return;
}

int MNNImpl::Evaluate(const std::vector<int> &tokens, std::string &intent) {
  if (!(model_type_ == MNN_MODEL_QUARTZNET ||
        model_type_ == MNN_MODEL_QUARTZNET_CNNATTN ||
      model_type_ == MNN_MODEL_QUARTZNET_REGEX)) {
    std::cout << "Model type: " << model_type_ << std::endl;
    return -1;
  }
  float intent_value= 0.0f;
  std::vector<int> slots;
  int intent_res = speechengine::CNNAttention::IntentSlotInfer(
                    model_info_list_[1], tokens, intent_value,
                    slots);
  return intent_res;
}
#endif
}  // End of namespace speechengine
