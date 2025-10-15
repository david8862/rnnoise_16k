// Reference from:
// https://github.com/YongyuG/rnnoise_16k/blob/master/main.c
// https://blog.csdn.net/ericbar/article/details/79567108
//
// build & run with following cmd:
// $ g++ -Wall -O2 -o rnnvad_postprocess_sample rnnvad_postprocess_sample.cpp -I<header file path> -L<lib file path> -lrnnoise -lm
// $ ./rnnvad_postprocess_sample -h
// Usage: rnnvad_postprocess_sample
// --input_file, -i: input raw audio file. default: 'input.wav'
// --chunk_size,  -c: audio chunk size to read every time. default: 640
// --output_file, -o: output txt file for voice segment start & stop time (in seconds). default: output.txt
//
// $ ./rnnvad_postprocess_sample -i vad_input.wav -o vad_output.txt
//
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <assert.h>
#include <vector>

#include <rnnoise.h>

#define MAX_STR_LEN 128
#define RNNOISE_FRAME_SIZE (160)  // RNNoise use hard-coded frame size (160 samples, 10ms)
#define RNNOISE_SAMPLE_RATE (16000)  // RNNoise use hard-coded sample rate


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VAD postprocess part

// VAD result structure
typedef struct vad_result
{
    bool speech_trigger = false;
    float real_bos = -1.0;
    float real_eos = -1.0;
} rnnvad_results;

// VAD postprocess param structure
typedef struct vad_param
{
    int min_speech_len = 12;                  // (10ms)*(12frames) = 120ms
    int speech_activate_windows_size = 12;
    int max_silence_len = 50;                 // (10ms)*(50frames) = 500ms
    int max_silence_len_multilan = 55;        // (10ms)*(55frames) = 550ms
    int min_speech_len_inside = 8;            //  (10ms)*(8frames) = 80ms
    std::vector<int> vadprob_array;
    bool speech_maybe_ending = false;
    float prob_threshold = 0.4;
    int accum_silence_len = 0;
    int accum_speech_begin_len = 0;
    int accum_speech_inside_len = 0;
    long total_frames = 0;                    // frames every 10ms or 160 samples
} rnnvad_params;


bool clear_inner_accum_after_end(rnnvad_params& vad_param)
{
    vad_param.vadprob_array.clear();

    vad_param.speech_maybe_ending = false;
    vad_param.accum_silence_len = 0;
    vad_param.accum_speech_begin_len = 0;
    vad_param.accum_speech_inside_len = 0;

    return true;
}

bool clear_inner_accum_when_find_begin(rnnvad_params& vad_param)
{
    vad_param.vadprob_array.clear();

    vad_param.speech_maybe_ending = false;
    vad_param.accum_silence_len = 0;
    vad_param.accum_speech_inside_len = 0;

    return true;
}

// smoothing per-frame VAD result
void vad_smoothing(rnnvad_params& vad_param, rnnvad_results& vad_result, int language)
{
    vad_param.total_frames += 1;

    if (vad_result.speech_trigger) {
        // ----------------speech is active --------------------
        assert(!vad_param.vadprob_array.empty());

        if (vad_param.speech_maybe_ending) {
            vad_param.accum_silence_len += 1;
            if(vad_param.vadprob_array.back() == 1) {
                vad_param.accum_speech_inside_len += 1;
                if(vad_param.accum_speech_inside_len >= vad_param.min_speech_len_inside) {
                    // trigger VAD action & recover all VAD begin related status
                    clear_inner_accum_when_find_begin(vad_param);
                    vad_result.speech_trigger = true;
                }
            }
            else if(vad_param.vadprob_array.back() == 0) {
                // ...
                vad_param.accum_speech_inside_len = 0;
            }
            if(vad_param.accum_silence_len >= ((language > 0 ? vad_param.max_silence_len_multilan : vad_param.max_silence_len) + vad_param.accum_speech_inside_len)) {
                // clear VAD action & recover all VAD end related status
                vad_result.speech_trigger = false;
                vad_result.real_eos = (vad_param.total_frames - vad_param.accum_silence_len) / 100.0;;
                clear_inner_accum_after_end(vad_param);
            }
        } // end of if vad_param.speech_maybe_ending
        else {
            // no end firstly in speech maybe ending // then compare length
            if(vad_param.vadprob_array.back() == 0) {
                vad_param.speech_maybe_ending = true;
                vad_param.accum_silence_len += 1;
            } else if(vad_param.vadprob_array.back() == 1) {
                vad_result.speech_trigger = true;
            }
            // vad_result.speech_trigger = true;
        }
    } // end of if speech trigger
    else {
        // ----------------speech is not active --------------------
        vad_result.real_bos = -1.0;
        if(static_cast<int> (vad_param.vadprob_array.size()) < vad_param.speech_activate_windows_size)
            return;

        // WINDOWS STRATEGY
        // HARD CONTRAINTS WITH CONTINUATION
        if(vad_param.vadprob_array.back() == 0) {
            vad_param.accum_speech_begin_len = 0;
            // return false;
        } else if(vad_param.vadprob_array.back() == 1) {
            vad_param.accum_speech_begin_len += 1;
            if (vad_param.accum_speech_begin_len >= vad_param.min_speech_len) {
                // trigger VAD active
                vad_result.speech_trigger = true;
                vad_param.accum_speech_begin_len = 0;
                vad_result.real_bos = (vad_param.total_frames - vad_param.min_speech_len) / 100.0;
            }
            // ----------------------------------------------------------------
        }
    }

    return;
}

// enqueue VAD probability into prob vector for smoothing
int add_prob(float vad_prob, float prob_threshold, std::vector<int>& vadprob_array)
{
    vadprob_array.emplace_back((vad_prob > prob_threshold) ? 1 : 0);
    return 0;
}

// VAD postprocess according to different language
void vad_postprocess(float vad_prob, rnnvad_params& vad_param, rnnvad_results& vad_result, int language)
{
    add_prob(vad_prob, vad_param.prob_threshold, vad_param.vadprob_array);
    vad_smoothing(vad_param, vad_result, language);

    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void show_progressbar(int progress, int total, int barWidth)
{
    float percentage = (float)progress / total;
    int filledLength = barWidth * percentage;

    printf("\r[");
    for (int i = 0; i < barWidth; i++) {
        if (i < filledLength) {
            printf("#");
        } else {
            printf(" ");
        }
    }
    printf("] %.1f%%", percentage * 100);

    fflush(stdout);
}


int rnnvad_postprocess_sample(char* input_file, int chunk_size, char* output_file)
{
    int ret;
    float vad_prob = 0.0;
    int vad_status, prev_vad_status=0;
    float voice_start_time = 0.0;
    float voice_stop_time = 0.0;


    if (chunk_size % RNNOISE_FRAME_SIZE != 0) {
        printf("WARNING: chunk_size is not multiple of RNNoise frame size %d, which will cause process issue!\n", RNNOISE_FRAME_SIZE);
    }

    // open input/output file
    FILE* fp_input = fopen(input_file, "rb");
    FILE* fp_output = fopen(output_file, "w+");

    // calculate audio sample number
    // 16-bit (2 bytes) per sample
    fseek(fp_input, 0, SEEK_END);
    long file_len = ftell(fp_input);
    long sample_num = file_len / 2;
    printf("sample number of input audio is %ld\n", sample_num);
    // re-direct file pointer back to head
    rewind(fp_input);

    // output voice timestamps txt file would be like:
    // cat output.txt
    // 16000               // sample rate
    // 8525.995            // total duration time (in seconds)
    // 197.949,199.200     // voice start/stop time (in seconds)
    // 200.159,202.829
    // ...
    //
    // write sample rate & audio duration time
    float audio_duration = (float)(sample_num) / 16000.0;
    fprintf(fp_output, "%d\n", 16000);
    fprintf(fp_output, "%.3f\n", audio_duration);

    // calculate audio chunk number based on chunk size
    long chunk_num = sample_num / chunk_size;
    //int tail_num = sample_num % chunk_size;
    printf("chunk number is %ld\n", chunk_num);

    // create RNNoise state context
    DenoiseState *st;
    st = rnnoise_create();

    // prepare data buffers
    short* short_buffer = (short*)calloc(chunk_size, sizeof(short));
    float* float_buffer = (float*)calloc(RNNOISE_FRAME_SIZE, sizeof(float));

    // prepare VAD postprocess data structure
    rnnvad_results vad_result;
    rnnvad_params vad_param;

    // read out wav header (usually 44 bytes) to bypass it
    fread(short_buffer, 1, 44, fp_input);

    // detect voice here
    for(long i=0; i < chunk_num; i++) {
        // read raw audio data from input file
        ret = fread(short_buffer, sizeof(short), chunk_size, fp_input);

        short* ptr_tmp_short = short_buffer;
        int loop_num = chunk_size / RNNOISE_FRAME_SIZE;

        for(int j=0; j < loop_num; j++) {
            // convert short data to float for rnnoise input
            for(int k=0; k < RNNOISE_FRAME_SIZE; k++) {
                float_buffer[k] = (float)(ptr_tmp_short[k]);
            }

            // process frame to get VAD probability
            vad_prob = rnnoise_process_frame(st, float_buffer, float_buffer);

            // VAD postprocess, language==0 means chinese
            vad_postprocess(vad_prob, vad_param, vad_result, 0);
            vad_status = (vad_result.speech_trigger) ? 1 : 0;

            // check VAD status to record speech start/stop time
            if (vad_status != prev_vad_status) {
                if ((prev_vad_status == 0) && (vad_status == 1)) {
                    // found VAD start point, record start time
                    voice_start_time = ((float)(i*chunk_size + j*RNNOISE_FRAME_SIZE)) / (float)(RNNOISE_SAMPLE_RATE);
                    prev_vad_status = vad_status;
                } else if ((prev_vad_status == 1) && (vad_status == 0)) {
                    // found VAD stop point, write the segment start & stop time into output file
                    voice_stop_time = ((float)(i*chunk_size + j*RNNOISE_FRAME_SIZE)) / (float)(RNNOISE_SAMPLE_RATE);
                    fprintf(fp_output, "%.3f,%.3f\n", voice_start_time, voice_stop_time);
                    // reset status
                    prev_vad_status = vad_status;
                    voice_start_time = 0.0;
                    voice_stop_time = 0.0;
                }
            }

            // move to next frame
            ptr_tmp_short += RNNOISE_FRAME_SIZE;
        }

        // show process bar
        int percentage = 100*i/chunk_num;
        show_progressbar(percentage, 100, 100);
    }

    // destroy RNNoise state context
    rnnoise_destroy(st);
    // release resources
    fclose(fp_input);
    fclose(fp_output);
    free(short_buffer);
    free(float_buffer);
    return 0;

}


void display_usage()
{
    printf("Usage: rnnvad_postprocess_sample\n" \
           "--input_file, -i: input raw audio file. default: 'input.wav'\n" \
           "--chunk_size, -c: audio chunk size to read every time. default: 640\n" \
           "--output_file, -o: output txt file for voice timestamps. default: output.txt\n" \
           "\n");
    return;
}


int main(int argc, char** argv)
{
    char input_file[MAX_STR_LEN] = "input.wav";
    int chunk_size = 640;
    char output_file[MAX_STR_LEN] = "output.txt";

    int c;
    while (1) {
        static struct option long_options[] = {
            {"input_file", required_argument, NULL, 'i'},
            {"chunk_size", required_argument, NULL, 'c'},
            {"output_file", required_argument, NULL, 'o'},
            {"help", no_argument, NULL, 'h'},
            {NULL, 0, NULL, 0}};

        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "c:hi:o:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'c':
                chunk_size = strtol(optarg, NULL, 10);
                break;
            case 'i':
                memset(input_file, 0, MAX_STR_LEN);
                strcpy(input_file, optarg);
                break;
            case 'o':
                memset(output_file, 0, MAX_STR_LEN);
                strcpy(output_file, optarg);
                break;
            case 'h':
            case '?':
            default:
                /* getopt_long already printed an error message. */
                display_usage();
                exit(-1);
        }
    }

    printf("NOTE: RNNoise lib only support single channel, 16k sample rate, 16-bit audio data!\n");
    rnnvad_postprocess_sample(input_file, chunk_size, output_file);

    printf("\nProcess finished.\n");
    return 0;
}
