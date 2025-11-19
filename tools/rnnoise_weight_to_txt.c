// -----------------------------------------------------------------------------
// Tool to dump RNNoise/RNNVad model weights from librnnvad.a to txt file, e.g. weight.txt
// for further .h5 model dump
//
// Reference:
// https://blog.csdn.net/lftxd1/article/details/123927528
//
// Step for dump:
// $ ar x speechengine/third_party/x86_64/rr_vad/rnnvad/librnnvad.a rnn_data.o
// $ objdump -t rnn_data.o
// $ gcc rnnoise_weight_to_txt.c rnn_data.o -o rnnoise_weight_to_txt
// $ ./rnnoise_weight_to_txt -h
// Usage: rnnoise_weight_to_txt
// --output_file, -o: output txt file for rnnoise model weights. default: weights.txt
//
// $ ./rnnoise_weight_to_txt -o weights.txt
//
//
// Then you can use "training_new/tools/model_converter/txt_to_keras.py" to generate keras
// .h5 pre-trained model weights file with the txt weights file:
//
// $ cd training_new/tools/model_converter/
// $ python txt_to_keras.py -h
// usage: txt_to_keras.py [-h] [--txt_file TXT_FILE]
//                             [--bands_num BANDS_NUM]
//                             [--delta_ceps_num DELTA_CEPS_NUM]
//                             [--keras_model_file KERAS_MODEL_FILE]
//
// Fulfill RNNoise/RNNVad model weights from txt model weights file
//
// options:
//   -h, --help            show this help message and exit
//   --txt_file TXT_FILE   input txt weights file, default=weights.txt
//   --bands_num BANDS_NUM
//                         number of bands, default=18
//   --delta_ceps_num DELTA_CEPS_NUM
//                         number of delta ceps, default=6
//   --keras_model_file KERAS_MODEL_FILE
//                         output .h5 keras model file, default=pretrained_weights.h5
//
// $ python txt_to_keras.py --txt_file=../../../tools/weights.txt
// txt to keras model: 100%|███████████████████████████████████████████████| 9/9 [00:00<00:00, 273.85it/s]
//
// Done. model weights has been saved to pre-trained model file pretrained_weights.h5
// -----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

#include "../src/rnn.h"
#include "../src/rnn_data.h"


int dump_array(const char* name, const rnn_weight *data, int size, FILE* fp) {
    fprintf(fp, "%s %d\n", name, size);

    for(int i=0;i < size; i++) {
        fprintf(fp, " %d,", (int)(data[i]));
        //if(i > 0 && i % 8 == 0)
        //    fprintf(fp, "\n");
    }

    return 0;
}


int rnnoise_weight_to_txt(char* output_file)
{
    FILE* fp = fopen(output_file, "w+");

    // dump weights of input_dense to txt
    dump_array("input_dense_weights", input_dense.input_weights, input_dense.nb_inputs * input_dense.nb_neurons, fp);
    fprintf(fp, "\n\n");
    dump_array("input_dense_bias", input_dense.bias, input_dense.nb_neurons, fp);
    fprintf(fp, "\n\n");

    //printf("input_dense.nb_inputs = %d\n", input_dense.nb_inputs);
    //printf("input_dense.nb_neurons = %d\n", input_dense.nb_neurons);
    //printf("input_dense.activataon = %d\n", input_dense.activation);

    // dump weights of vad_gru to txt, each GRU unit has 3 weight params
    dump_array("vad_gru_weights", vad_gru.input_weights, vad_gru.nb_inputs * vad_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");
    dump_array("vad_gru_recurrent_weights", vad_gru.recurrent_weights, vad_gru.nb_neurons * vad_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");
    dump_array("vad_gru_bias", vad_gru.bias, vad_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");

    // dump weights of noise_gru to txt
    dump_array("noise_gru_weights", noise_gru.input_weights, noise_gru.nb_inputs * noise_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");
    dump_array("noise_gru_recurrent_weights", noise_gru.recurrent_weights, noise_gru.nb_neurons * noise_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");
    dump_array("noise_gru_bias", noise_gru.bias, noise_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");

    // dump weights of denoise_gru to txt
    dump_array("denoise_gru_weights", denoise_gru.input_weights, denoise_gru.nb_inputs * denoise_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");
    dump_array("denoise_gru_recurrent_weights", denoise_gru.recurrent_weights, denoise_gru.nb_neurons * denoise_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");
    dump_array("denoise_gru_bias", denoise_gru.bias, denoise_gru.nb_neurons * 3, fp);
    fprintf(fp, "\n\n");

    // dump weights of denoise_output to txt
    dump_array("denoise_output_weights", denoise_output.input_weights, denoise_output.nb_inputs * denoise_output.nb_neurons, fp);
    fprintf(fp, "\n\n");
    dump_array("denoise_output_bias", denoise_output.bias, denoise_output.nb_neurons, fp);
    fprintf(fp, "\n\n");

    // dump weights of vad_output to txt
    dump_array("vad_output_weights", vad_output.input_weights, vad_output.nb_inputs * vad_output.nb_neurons, fp);
    fprintf(fp, "\n\n");
    dump_array("vad_output_bias", vad_output.bias, vad_output.nb_neurons, fp);
    fprintf(fp, "\n\n");

    printf("\nDone. All the weights data has been dumped to %s\n", output_file);
    fclose(fp);
    return 0;
}

#define MAX_STR_LEN 128

void display_usage()
{
    printf("Usage: rnnoise_weight_to_txt\n" \
           "--output_file, -o: output txt file for rnnoise model weights. default: weights.txt\n" \
           "\n");
    return;
}


int main(int argc, char** argv)
{
    char output_file[MAX_STR_LEN] = "weights.txt";

    int c;
    while (1) {
        static struct option long_options[] = {
            {"output_file", required_argument, NULL, 'o'},
            {"help", no_argument, NULL, 'h'},
            {NULL, 0, NULL, 0}};

        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "ho:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
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

    rnnoise_weight_to_txt(output_file);

    return 0;
}
