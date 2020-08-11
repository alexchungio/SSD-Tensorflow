#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : read_tfrecord.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/4 下午3:06
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.python_io import tf_record_iterator
from libs.tf_extend import tf_utils


from libs.box_utils import draw_box_in_image
from libs.config import cfgs
from data.preprocessing.ssd_preprocessing import preprocess_image

# origin_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_split/val'
tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd'


def dataset_tfrecord(dataset_dir, split_name='train', batch_size=32, anchor_encoder_fn=None, batch_shape = None,
                     num_epochs=None, shuffle=True, num_threads=4, is_training=False):
    """
    parse tensor
    :param image_sample:
    :return:
    """
    # construct feature description
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/num_object': tf.FixedLenFeature([1], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/num_object': slim.tfexample_decoder.Tensor('image/object/num_object'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {}
    for name, pair in cfgs.VOC_LABELS.items():
        labels_to_names[pair[0]] = name

    dataset = slim.dataset.Dataset(
        data_sources=os.path.join(dataset_dir, '*'),
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=cfgs.NUM_SPLIT_DATA[split_name],
        items_to_descriptions=None,
        num_classes=cfgs.NUM_CLASS,
        labels_to_names=labels_to_names)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=cfgs.NUM_READER,
            common_queue_capacity=32 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs)

    [image, filename, shape, num_object, labels, bboxes, is_difficult] = provider.get(['image', 'filename', 'shape', 'object/num_object',
                                                                             'object/label','object/bbox',
                                                                             'object/difficult'])

    image, labels, bboxes = image_process(image, bboxes, labels, is_training=is_training)
    # Encode groundtruth labels and bboxes.
    gt_labels, gt_bbox, gt_scores = anchor_encoder_fn(labels, bboxes)

    data_batch =  tf.train.batch(tf_utils.reshape_list([image, filename, shape, gt_labels, gt_bbox, gt_scores]),
                         dynamic_pad=False,
                         batch_size=batch_size,
                         allow_smaller_final_batch=(not is_training),
                         num_threads=num_threads,
                         capacity=5 * batch_size)
    image, filename, shape, gt_labels, gt_bbox, gt_scores = tf_utils.reshape_list(data_batch, batch_shape)
    return image, filename, shape, gt_labels, gt_bbox, gt_scores


def image_process(image, gtboxes, labels, is_training=False):
    """
    image process
    :param image:
    :param gtboxes_and_label:
    :param shortside_len:
    :return:
    """

    if is_training:
        # if all is difficult, then keep the first one
        # is_difficult_mask = tf.cond(tf.count_nonzero(is_difficult, dtype=tf.int32) < tf.shape(is_difficult)[0],
        #                            lambda: is_difficult < tf.ones_like(is_difficult),
        #                            lambda: tf.one_hot(0, tf.shape(is_difficult)[0], on_value=True, off_value=False,
        #                                               dtype=tf.bool))
        #
        # gtboxes = tf.boolean_mask(gtboxes, is_difficult_mask)
        # labels = tf.boolean_mask(labels, is_difficult_mask)
        image, glabels, gbboxes = preprocess_image(image, labels, gtboxes, data_format=cfgs.DATA_FORMAT,
                                                   out_shape=cfgs.TRAIN_SIZE, is_training=True)

    else:
        image, glabels, gbboxes = preprocess_image(image, labels, gtboxes, data_format=cfgs.DATA_FORMAT,
                                                   out_shape=cfgs.EVAL_SIZE, is_training=False)

    return image, glabels, gbboxes


def get_num_samples(record_dir):
    """
    get tfrecord numbers
    :param record_file:
    :return:
    """

    # check record file format
    record_list = glob.glob(os.path.join(record_dir, '*.record'))

    num_samples = 0
    for record_file in record_list:
        for record in tf_record_iterator(record_file):
            num_samples += 1
    return num_samples


if __name__ == "__main__":

    print('number samples: {0}'.format(get_num_samples(tfrecord_dir)))
    # create local and global variables initializer group
    # image, filename, gtboxes_and_label, num_objects = reader_tfrecord(record_file=tfrecord_dir,
    #                                                                   shortside_len=IMG_SHORT_SIDE_LEN,
    #                                                                   is_training=True)
    image_batch, filename_batch, shape_batch, labels_batch, bboxes_batch, score_batch = dataset_tfrecord(tfrecord_dir, split_name='train',
                                                                                            batch_size=1, is_training=True)
    # gtboxes_and_label_tensor = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    # gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=image_batch,
    #                                                                boxes=gtboxes_and_label_tensor[:, :-1],
    #                                                                labels=gtboxes_and_label_tensor[:, -1])
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)
        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            if not coord.should_stop():
                img_name, image, bboxes, labels = sess.run([filename_batch, image_batch,  bboxes_batch,
                                                                labels_batch])

                plt.imshow(image[0])
                # print(filename[0])
                print(bboxes[0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()
        # waiting all threads safely exit
        coord.join(threads)
        sess.close()