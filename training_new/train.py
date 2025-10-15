#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train RNNoise model
"""
import os, sys, argparse, time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TerminateOnNaN
import tensorflow.keras.backend as K
import tensorflow as tf

from rnnoise.model import get_model
from rnnoise.loss import denoise_loss, vad_loss, msse
from rnnoise.data import get_train_data
from common.utils import optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import CheckpointCleanCallBack

optimize_tf_gpu(tf, K)


def train(args):
    log_dir = 'logs/000'
    os.makedirs(log_dir, exist_ok=True)

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-denoise_output_loss{denoise_output_loss:.3f}-vad_output_loss{vad_output_loss:.3f}.h5'),
        monitor='denoise_output_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    #reduce_lr = ReduceLROnPlateau(monitor='denoise_output_loss', mode='min', factor=0.5, patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='denoise_output_loss', mode='min', min_delta=0, patience=50, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_keep=5)
    #learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    #lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])

    callbacks = [logging, checkpoint, early_stopping, terminate_on_nan, checkpoint_clean]

    # prepare train dataset
    x_train, y_train, vad_train = get_train_data(args.train_data_file, args.window_size)

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    # get train model
    model = get_model(weights_path=args.weights_path)
    model.summary()

    model.compile(loss=[denoise_loss, vad_loss],
                  metrics=[msse],
                  optimizer=optimizer, loss_weights=[10, 0.5])

    model.fit(x_train, [y_train, vad_train],
              batch_size=args.batch_size,
              epochs=args.total_epoch,
              validation_split=args.val_split,
              callbacks=callbacks)


    # Freeze backbone part for transfer learning
    #for i in range(backbone_len):
    #    model.layers[i].trainable = False
    #print('Freeze the first {} layers of total {} layers.'.format(backbone_len, len(model.layers)))
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    #initial_epoch = args.init_epoch
    #epochs = initial_epoch + args.transfer_epoch
    #print("Transfer training stage")
    #print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(train_generator.samples, val_generator.samples, args.batch_size, args.model_input_shape))
    #model.fit_generator(train_generator,
    #                    steps_per_epoch=train_generator.samples // args.batch_size,
    #                    validation_data=val_generator,
    #                    validation_steps=val_generator.samples // args.batch_size,
    #                    epochs=epochs,
    #                    initial_epoch=initial_epoch,
    #                    #verbose=1,
    #                    workers=1,
    #                    use_multiprocessing=False,
    #                    max_queue_size=10,
    #                    callbacks=callbacks)

    # Wait 2 seconds for next stage
    #time.sleep(2)

    #if args.decay_type:
        # rebuild optimizer to apply learning rate decay, only after
        # unfreeze all layers
    #    callbacks.remove(reduce_lr)
    #    steps_per_epoch = max(1, train_generator.samples//args.batch_size)
    #    decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch - args.transfer_epoch)
    #    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=args.decay_type, decay_steps=decay_steps)


    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    #for i in range(len(model.layers)):
    #    model.layers[i].trainable = True
    #print("Unfreeze and continue training, to fine-tune.")
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    #model.fit_generator(train_generator,
    #                    steps_per_epoch=train_generator.samples // args.batch_size,
    #                    validation_data=val_generator,
    #                    validation_steps=val_generator.samples // args.batch_size,
    #                    epochs=args.total_epoch,
    #                    initial_epoch=epochs,
                        #verbose=1,
    #                    workers=1,
    #                    use_multiprocessing=False,
    #                    max_queue_size=10,
    #                    callbacks=callbacks)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='train RNNoise model')
    # Model definition options
    parser.add_argument('--window_size', type=int, required=False, default=2000,
        help="input window size, default=%(default)s")
    #parser.add_argument('--model_type', type=str, required=False, default='mobilenetv2',
    #    help='backbone model type: mobilenetv3/v2/simple_cnn, default=%(default)s')
    #parser.add_argument('--model_input_shape', type=str, required=False, default='224x224',
    #    help = "model image input shape as <height>x<width>, default=%(default)s")
    #parser.add_argument('--head_conv_channel', type=int, required=False, default=128,
    #    help = "channel number for head part convolution, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--train_data_file', type=str, required=True,
        help='h5 file for training dataset')
    parser.add_argument('--val_split', type=float, required=False, default=0.1,
        help="validation data persentage in dataset, default=%(default)s")
    #parser.add_argument('--val_data_path', type=str, required=True,
    #    help='path to validation image dataset')
    #parser.add_argument('--classes_path', type=str, required=False,
    #    help='path to classes definition', default=None)

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=64,
        help = "batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")

    #parser.add_argument('--init_epoch', type=int,required=False, default=0,
    #    help = "Initial training epochs for fine tune training, default=%(default)s")
    #parser.add_argument('--transfer_epoch', type=int, required=False, default=5,
    #    help = "Transfer training (from Imagenet) stage epochs, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=100,
        help = "Total training epochs, default=%(default)s")
    #parser.add_argument('--gpu_num', type=int, required=False, default=1,
    #    help='Number of GPU to use, default=%(default)s')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
