#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/5 下午5:52
# @ Software   : PyCharm
#-------------------------------------------------------fa

import os
import time
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from data.pascal.read_tfrecord import dataset_tfrecord
from libs.nets.ssd_300_vgg import SSDNet
from libs.tf_extend import tf_utils
from utils.tools import makedir
from libs.deployment import model_deploy


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def train():

    # -----------------step 1  Get the SSD network and its anchors.----------------
    SSDNet.default_params._replace(num_classes=cfgs.NUM_CLASS+1)
    # construct ssd net
    ssd_net = SSDNet()
    ssd_shape = ssd_net.params.img_shape
    global_step = ssd_net.global_step
    ssd_anchors = ssd_net.make_anchors(ssd_shape)
    # ----------------- step 2 Create a dataset provider and batches.---------------

    with tf.name_scope(cfgs.DATASET_NAME + '_data_provider'):

        anchor_encoder_fn = lambda gt_labels, gb_bboxes: ssd_net.bboxes_encode(gt_labels, gb_bboxes, ssd_anchors,
                                                                               scope="anchor_encode")
        batch_shape = [1] * 3 + [len(ssd_anchors)] * 3
        image_batch, filename_batch, shape_batch, labels_batch, bboxes_batch, scores_batch = dataset_tfrecord(
            dataset_dir=cfgs.TFRECORD_DIR, split_name='train', batch_size=cfgs.BATCH_SIZE,
            anchor_encoder_fn=anchor_encoder_fn, batch_shape=batch_shape, num_threads=cfgs.NUM_THREADS,
            is_training=True)

        # batch_queue = slim.prefetch_queue.prefetch_queue(
        #     tf_utils.reshape_list([image_batch, filename_batch, shape_batch, labels_batch, bboxes_batch, scores_batch]),
        #     capacity=2)
        #
        # image_batch, filename_batch, shape_batch, labels_batch, bboxes_batch, scores_batch = \
        #     tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)
    # -------------------step 3 construct foward network-----------------------------

    # Construct SSD network.
    arg_scope = ssd_net.arg_scope(weight_decay=cfgs.WEIGHT_DECAY, data_format=cfgs.DATA_FORMAT)
    with slim.arg_scope(arg_scope):
        predictions, localisations, logits, end_points = ssd_net.net(image_batch, is_training=True)
    # Add loss function.
    ssd_net.losses(logits, localisations,
                   labels_batch, bboxes_batch, scores_batch,
                   match_threshold=cfgs.MATCH_THRESHOLD,
                   negative_ratio=cfgs.NEGATIVE_RATIO,
                   alpha=cfgs.LOSS_ALPHA,
                   label_smoothing=cfgs.LABELS_SMOOTH)

    # Gather initial summaries.
    # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # =================================================================== #
    # Add summaries from first clone.
    # =================================================================== #
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Add summaries for end_points.
    # Add summaries for losses and extra losses.
    # for loss in tf.get_collection(tf.GraphKeys.LOSSES):
    #     tf.summary.scalar(loss.op.name, loss)

    # Add summaries for variables.
    for variable in slim.get_model_variables():
       tf.summary.histogram(variable.op.name, variable)

    # =================================================================== #
    # Configure the moving averages.
    # =================================================================== #
    if cfgs.MOVING_AVERATE_DECAY:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            cfgs.MOVING_AVERATE_DECAY, ssd_net.global_step)
    else:
        moving_average_variables, variable_averages = None, None

    # =================================================================== #
    # Configure the optimization procedure.
    # =================================================================== #

    learning_rate = ssd_net.learning_rate(boundaries=[cfgs.WARM_UP_STEP, cfgs.DECAY_STEP[0], cfgs.DECAY_STEP[1]],
                                          rates=[cfgs.WARM_UP_LEARING_RATE, cfgs.LEARING_RATE_BASE,
                                                 cfgs.LEARING_RATE_BASE / 10., cfgs.LEARING_RATE_BASE / 100.],
                                          global_step=ssd_net.global_step,
                                          warmup=True)

    truncated_learning_rate = tf.maximum(learning_rate,  tf.constant(cfgs.END_LEARNING_RATE, dtype=learning_rate.dtype), name='learning_rate')

    tf.summary.scalar('learning_rate', truncated_learning_rate)
    #-
    optimizer = tf.train.MomentumOptimizer(truncated_learning_rate, momentum=cfgs.MOMENTUM)
    if cfgs.MOVING_AVERATE_DECAY:
        # Update ops executed locally by trainer.
        update_ops.append(variable_averages.apply(moving_average_variables))


    # and returns a train_tensor and summary_op
    total_loss, clones_gradients = ssd_net.optimize_gradient(optimizer=optimizer)
    # Add total_loss to summary.
    tf.summary.scalar('total_loss', total_loss)

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    summary_op = tf.summary.merge_all()

    # =================================================================== #
    # Kicks off the training.
    # =================================================================== #
    saver = tf.train.Saver(max_to_keep=30,
                           keep_checkpoint_every_n_hours=2.0,
                           write_version=2)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfgs.GPU_MEMORY_FRACTION,
                                allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False,
                            gpu_options=gpu_options)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # if not restorer is None:
        #     restorer.restore(sess, save_path=restore_ckpt)
        restore_ckpt = ssd_net.restore_ckpt(sess)
        print('*' * 80 + '\nSuccessful restore model from {0}\n'.format(restore_ckpt) + '*' * 80)
        # model_variables = slim.get_model_variables()
        # for var in model_variables:
        #     print(var.name, var.shape)
        # build summary write
        summary_writer = tf.summary.FileWriter(cfgs.SUMMARY_PATH, graph=sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        # ++++++++++++++++++++++++++++++++++++++++start training+++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            if not coord.should_stop():
                for step in range(cfgs.MAX_ITERATION):

                    # image, labels, bboxes, scores = \
                    #     sess.run([image_batch, labels_batch, bboxes_batch, scores_batch])
                    # feed_dict = ssd_net.fill_feed_dict(image, labels, bboxes, scores)
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                    if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                        _, globalStep = sess.run([train_op, global_step])
                    else:
                        if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                            start_time = time.time()

                            _, globalStep, totalLoss = sess.run([train_op, global_step, total_loss])

                            end_time = time.time()
                            print(""" {}: step {}\t | total_loss:{} |\t per_cost_time:{}s""" \
                                  .format(training_time, globalStep,  totalLoss,(end_time - start_time)))
                        else:
                            if step % cfgs.SMRY_ITER == 0:
                                _, globalStep, summary_str = sess.run([train_op, global_step, summary_op])
                                summary_writer.add_summary(summary_str, globalStep)
                                summary_writer.flush()

                    if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):
                        save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                        makedir(save_dir)
                        save_ckpt = os.path.join(save_dir, 'voc_' + str(globalStep) + '_model.ckpt')
                        saver.save(sess, save_ckpt)
                        print(' weights had been saved')

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
            print('all threads are asked to stop!')


if __name__ == "__main__":
    train()