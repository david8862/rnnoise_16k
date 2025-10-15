/*
 * postprocess.h
 *
 *  Created on: 2023年10月28日
 *      Author: zhaojianping
 */

#ifndef INCLUDE_POSTPROCESS_H_
#define INCLUDE_POSTPROCESS_H_
#include <cassert>
#include <vector>
typedef struct VADSTATE
{
	bool speech_trigger = false;
	float real_bos = -1.0;
	float real_eos = -1.0;
} RNNVAD_RESULTS;

class RNNVAD_POSTPROCESS{
  public:

    RNNVAD_POSTPROCESS(){}
    ~RNNVAD_POSTPROCESS(){}

  private:
    int min_speech_len = 12; // * 10 = 150ms
    int speech_activate_windows_size = 12;
    int max_silence_len = 50; //* 10 =  500ms
    int max_silence_len_multilan = 55; //* 10 =  500ms
    int min_speech_len_inside = 8; // * 10 = 100ms
    std::vector<int> vadprob_array;
    bool speech_active = false;
    bool speech_maybe_ending = false;
    float prob_tld = 0.4;
    int accum_silence_len = 0;
    int accum_speech_begin_len = 0;
    int accum_speech_inside_len = 0;
    long total_frms = 0; // frms every 10ms or 160
    RNNVAD_RESULTS vadstatus;

  public:
    RNNVAD_RESULTS do_vadsmoothing(int language)
    {
      total_frms+=1;
      if (vadstatus.speech_trigger)
      {
        assert(!vadprob_array.empty());
        if (speech_maybe_ending)
        {
          accum_silence_len += 1;
          if(vadprob_array.back() == 1)
          {
            accum_speech_inside_len+=1;
            if(accum_speech_inside_len >= min_speech_len_inside)
            {
              // recover all vad end status
              clear_inner_accum_when_find_begin();
              vadstatus.speech_trigger = true;
            }
          }
          else if(vadprob_array.back() == 0)
          {
              // ...
              accum_speech_inside_len = 0;
          }
          if(accum_silence_len >= ((language > 0 ? max_silence_len_multilan : max_silence_len) + accum_speech_inside_len))
          {
            vadstatus.speech_trigger = false;
            vadstatus.real_eos = (total_frms - accum_silence_len) / 100.0;;
            clear_inner_accum_after_end();
          }
        } // end of if speech_maybe_ending
        else
        {
          // no end firstly in speech maybe ending // then compare length
          if(vadprob_array.back() == 0)
          {
            speech_maybe_ending = true;
            accum_silence_len += 1;
          }
         else if(vadprob_array.back() == 1)
         {
             vadstatus.speech_trigger = true;
         }
//          vadstatus.speech_trigger = true;
        }
      } // end of if speech trigger
      else{
        // ----------------speech is not active --------------------
    	 vadstatus.real_bos = -1.0;
        if(static_cast<int> (vadprob_array.size()) < speech_activate_windows_size) return vadstatus;
        // WINDOWS STRATEGY
        // HARD CONTRAINTS WITH CONTINUATION
        if(vadprob_array.back() == 0)
        {
          accum_speech_begin_len = 0;
//          return false;
        }
        else if(vadprob_array.back() == 1)
        {
          accum_speech_begin_len += 1;
          if (accum_speech_begin_len >= min_speech_len)
          {
        	  vadstatus.speech_trigger = true;
            accum_speech_begin_len = 0;
            vadstatus.real_bos = (total_frms - min_speech_len) / 100.0;
            // speech active
          }
        // ----------------------- -----------------------------------------
        }
      }
      return vadstatus;
    }
    int add_prob(float prob)
    {
      vadprob_array.emplace_back((prob > prob_tld)?1:0);
      return 0;
    }
    bool clear_vad_status()
    {
      return vadprob_array.empty();
    }
    bool get_vad_status()
    {
      return speech_active;
    }
    void set_vad_thresold(float threshold)
    {
      // assert(threshold >= 0.0 && thread_local < 1.0);
      prob_tld = threshold;
    }
    float get_vad_thresold()
    {
      // assert(threshold >= 0.0 && thread_local < 1.0);
      return prob_tld;
    }
    RNNVAD_RESULTS do_postprocess(float prob, int language)
    {
      add_prob(prob);
      return do_vadsmoothing(language);
    }
    bool clear_inner_accum_after_end()
    {
      vadprob_array.clear();

      speech_maybe_ending = false;
      accum_silence_len = 0;
      accum_speech_begin_len = 0;
      accum_speech_inside_len = 0;
      return true;
    }

    bool clear_inner_accum_when_find_begin()
    {
      vadprob_array.clear();
      speech_maybe_ending = false;
      accum_silence_len = 0;
      accum_speech_inside_len = 0;
      return true;
    }

};


#endif /* INCLUDE_POSTPROCESS_H_ */
/*
 //    RNNVAD_POSTPROCESS(float prob_tld,int min_speech_len, int max_silence_len, \
//    		int min_speech_len_inside):prob_tld(prob_tld),(min_speech_len), \
//			max_silence_len(max_silence_len), min_speech_len_inside(min_speech_len_inside){}

 */
