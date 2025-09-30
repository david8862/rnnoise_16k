// Reference from:
// https://github.com/YongyuG/rnnoise_16k/blob/master/main.c
//
// build & run with following cmd:
// $ gcc -Wall -O2 -o rnnvad_sample rnnvad_sample.c -I<header file path> -L<lib file path> -lrnnoise -lm
// $ ./rnnvad_sample -h
// Usage: rnnvad_sample
// --input_file, -i: input raw audio file. default: 'input.wav'
// --chunk_size,  -c: audio chunk size to read every time. default: 640
// --vad_threshold, -t: threshold for vad probability. default: 0.5
// --output_file, -o: output txt file for voice segment start & stop time (in seconds). default: output.txt
//
// $ ./rnnvad_sample -i vad_input.wav -o vad_output.txt
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <rnnoise.h>

#define MAX_STR_LEN 128
#define RNNOISE_FRAME_SIZE (160)  // RNNoise use hard-coded frame size
#define RNNOISE_SAMPLE_RATE (16000)  // RNNoise use hard-coded sample rate


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

int rnnvad_sample(char* input_file, int chunk_size, float vad_threshold, char* output_file)
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

    // calculate audio chunk number based on chunk size
    long chunk_num = sample_num / chunk_size;
    int tail_num = sample_num % chunk_size;
    printf("chunk number is %ld\n", chunk_num);

    // create RNNoise state context
    DenoiseState *st;
    st = rnnoise_create();

    // prepare data buffers
    short* short_buffer = (short*)calloc(chunk_size, sizeof(short));
    float* float_buffer = (float*)calloc(RNNOISE_FRAME_SIZE, sizeof(float));

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

            // check VAD probability with threshold to confirm status
            vad_status = (vad_prob > vad_threshold) ? 1 : 0;
            if (vad_status != prev_vad_status) {
                if ((prev_vad_status == 0) && (vad_status == 1)) {
                    // found VAD start point, record start time
                    voice_start_time = ((float)(i*chunk_size + j*RNNOISE_FRAME_SIZE)) / (float)(RNNOISE_SAMPLE_RATE);
                    prev_vad_status = vad_status;
                } else if ((prev_vad_status == 1) && (vad_status == 0)) {
                    // found VAD stop point, write the segment start & stop time into output file
                    voice_stop_time = ((float)(i*chunk_size + j*RNNOISE_FRAME_SIZE)) / (float)(RNNOISE_SAMPLE_RATE);
                    fprintf(fp_output, "%f,%f\n", voice_start_time, voice_stop_time);
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
    printf("Usage: rnnvad_demo\n" \
           "--input_file, -i: input raw audio file. default: 'input.wav'\n" \
           "--chunk_size, -c: audio chunk size to read every time. default: 640\n" \
           "--vad_threshold, -t: threshold for vad probability. default: 0.5\n" \
           "--output_file, -o: output txt file for voice segment start & stop time (in seconds). default: output.txt\n" \
           "\n");
    return;
}


int main(int argc, char** argv)
{
    char input_file[MAX_STR_LEN] = "input.wav";
    int chunk_size = 640;
    float vad_threshold = 0.5;
    char output_file[MAX_STR_LEN] = "output.txt";

    int c;
    while (1) {
        static struct option long_options[] = {
            {"input_file", required_argument, NULL, 'i'},
            {"chunk_size", required_argument, NULL, 'c'},
            {"vad_threshold", required_argument, NULL, 't'},
            {"output_file", required_argument, NULL, 'o'},
            {"help", no_argument, NULL, 'h'},
            {NULL, 0, NULL, 0}};

        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "c:hi:o:t:", long_options, &option_index);

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
            case 't':
                vad_threshold = strtof(optarg, NULL);
                break;
            case 'h':
            case '?':
            default:
                /* getopt_long already printed an error message. */
                display_usage();
                exit(-1);
        }
    }

    printf("NOTE: RNNoise lib only support 16k sample rate, 16-bit audio data!\n");
    rnnvad_sample(input_file, chunk_size, vad_threshold, output_file);

    printf("\nProcess finished.\n");
    return 0;
}
