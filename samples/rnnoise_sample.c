// Reference from:
// https://github.com/YongyuG/rnnoise_16k/blob/master/main.c
//
// build & run with following cmd:
// $ gcc -Wall -O2 -o rnnoise_sample rnnoise_sample.c -I<header file path> -L<lib file path> -lrnnoise -lm
// $ ./rnnoise_sample -h
// Usage: rnnoise_sample
// --input_file, -i: input raw audio file. default: 'input.wav'
// --chunk_size,  -c: audio chunk size to read every time. default: 640
// --output_file, -o: output pcm file for denoised audio. default: output.pcm
//
// $ ./rnnoise_sample -i denoise_input.wav -o denoise_output.pcm
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

#include <rnnoise.h>

#define MAX_STR_LEN 128
#define RNNOISE_FRAME_SIZE (160)  // RNNoise use hard-coded frame size


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

int rnnoise_sample(char* input_file, int chunk_size, char* output_file)
{
    int ret;

    if (chunk_size % RNNOISE_FRAME_SIZE != 0) {
        printf("WARNING: chunk_size is not multiple of RNNoise frame size %d, which will cause process issue!\n", RNNOISE_FRAME_SIZE);
    }

    // open input/output file
    FILE* fp_input = fopen(input_file, "rb");
    FILE* fp_output = fopen(output_file, "wb");

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
    short* output_buffer = (short*)calloc(RNNOISE_FRAME_SIZE, sizeof(short));

    // read out wav header (usually 44 bytes) to bypass it
    fread(short_buffer, 1, 44, fp_input);

    // denoise here
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

            // process frame to denoise
            rnnoise_process_frame(st, float_buffer, float_buffer);

            // convert output float data back to short for write file
            for(int k=0; k < RNNOISE_FRAME_SIZE; k++) {
                output_buffer[k] = (short)(float_buffer[k]);
            }

            // save denoised audio to output file
            fwrite(output_buffer, sizeof(short), RNNOISE_FRAME_SIZE, fp_output);

            // move to next frame
            ptr_tmp_short += RNNOISE_FRAME_SIZE;
        }

        // show process bar
        int percentage = 100*i/chunk_num;
        show_progressbar(percentage, 100, 100);
    }
    // just save some data as tail, to align output length with input
    fwrite(short_buffer, sizeof(short), tail_num, fp_output);

    // destroy RNNoise state context
    rnnoise_destroy(st);
    // release resources
    fclose(fp_input);
    fclose(fp_output);
    free(short_buffer);
    free(float_buffer);
    free(output_buffer);
    return 0;

}


void display_usage()
{
    printf("Usage: rnnoise_sample\n" \
           "--input_file, -i: input raw audio file. default: 'input.wav'\n" \
           "--chunk_size,  -c: audio chunk size to read every time. default: 640\n" \
           "--output_file, -o: output pcm file for denoised audio. default: output.pcm\n" \
           "\n");
    return;
}


int main(int argc, char** argv)
{
    char input_file[MAX_STR_LEN] = "input.wav";
    int chunk_size = 640;
    char output_file[MAX_STR_LEN] = "output.pcm";

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
    rnnoise_sample(input_file, chunk_size, output_file);

    printf("\nProcess finished.\n");
    return 0;
}

