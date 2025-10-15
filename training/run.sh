
#../src/denoise_training_gao /ssd/gaoyongyu/data/speech_dir /ssd/gaoyongyu/data/noise_dir mixed.wav > training_16k_v3.f32
../src/denoise_train_data_creator -s /ssd/gaoyongyu/data/speech_dir -n /ssd/gaoyongyu/data/noise_dir -l 300000 -a mixed.wav -f training_16k_v3.f32

#python bin2hdf5.py training_16k_v3.f32 80000000 75 training_16k_v3.h5
python bin2hdf5.py --input_bin_file=training_16k_v3.f32 --matrix_shape=300000x75 --output_h5_file=training_16k_v3.h5

#python rnn_train_16k.py
python rnn_train_16k_test.py

python dump_rnn.py weights.hdf5 rnn_data.c rnn_data.rnnn
