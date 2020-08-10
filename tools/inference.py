#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/10 上午11:36
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2 as cv


from libs.config import cfgs
from libs.nets import ssd_vgg_300
import libs.box_utils.boxe_utils_np as np_methods
from data.preprocessing import ssd_preprocessing
from  libs.box_utils import draw_box_in_image
from utils.tools import makedir, view_bar


class ObjectInference():
    def __init__(self, net_shape=(300, 300), num_classes=21, ckpt_path=None, select_threshold=0.5, nms_threshold=0.45):
        self.net_shape = net_shape
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.ssd_net = ssd_vgg_300.SSDNet()
        self.data_format = cfgs.DATA_FORMAT
        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.make_anchors(net_shape)
        self.select_threshold = select_threshold
        self.nms_threshold = nms_threshold
        self.bbox_image = (0.0, 0.0, 1.0, 1.0)
        # self._R_MEAN = 123.68
        # self._G_MEAN = 116.779
        # self._B_MEAN = 103.939

    def exucute_detect(self, image_path, save_path):
        """
        execute object detect
        :param detect_net:
        :param image_path:
        :return:
        """
        input_image = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3), name='inputs_images')

        image_pre, labels_pre, bboxes_pre = self.image_process(input_image, img_shape=self.net_shape,
                                                               img_format=self.data_format)
        # expend dimension
        image_batch = tf.expand_dims(input=image_pre, axis=0)  # (1, None, None, 3)

        # img_shape = tf.shape(inputs_img)
        # load detect network
        reuse = True if 'ssd_net' in locals() else None
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
            detection_category, detection_bbox , _, _ = self.ssd_net.net(image_batch, is_training=False, reuse=reuse)

        # restore pretrain weight
        restorer = tf.train.Saver()

        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            if restorer is not None:
                restorer.restore(sess, self.ckpt_path)
                print('*'*80 +'\nSuccessful restore model from {0}\n'.format(self.ckpt_path) + '*'*80)

            # construct image path list
            format_list = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
            if os.path.isfile(image_path):
                image_name_list = [image_path]
            else:
                image_name_list = [img_name for img_name in os.listdir(image_path)
                              if img_name.endswith(format_list) and os.path.isfile(os.path.join(image_path, img_name))]

            assert len(image_name_list) != 0
            print("test_dir has no imgs there. Note that, we only support img format of {0}".format(format_list))
            #+++++++++++++++++++++++++++++++++++++start detect+++++++++++++++++++++++++++++++++++++++++++++++++++++=++
            makedir(save_path)
            fw = open(os.path.join(save_path, 'detect_bbox.txt'), 'w')

            for index, img_name in enumerate(image_name_list):

                detect_dict = {}
                bgr_img = cv.imread(os.path.join(image_path, img_name))
                rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB) # convert channel from BGR to RGB (cv is BGR)

                start_time = time.perf_counter()
                # image resize and white process
                # construct feed_dict
                # Run SSD network.]
                feed_dict = {input_image: rgb_img}
                image, category, bbox = sess.run([image_batch, detection_category, detection_bbox],
                                                               feed_dict=feed_dict)

                # Get classes and bboxes from the net outputs.
                rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(category, bbox, self.ssd_anchors,
                                                                          select_threshold=self.select_threshold,
                                                                          img_shape=self.net_shape,
                                                                          num_classes=self.num_classes, decode=True)

                rbboxes = np_methods.bboxes_clip(self.bbox_image, rbboxes)
                rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
                rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes,
                                                                   nms_threshold=self.nms_threshold)
                # Resize bboxes to original image shape. Note: useless for Resize.WARP!
                rbboxes = np_methods.bboxes_resize(self.bbox_image, rbboxes)
                end_time = time.perf_counter()

                rbboxes = np_methods.bboxes_recover(rbboxes, rgb_img)
                final_detections_img = draw_box_in_image.draw_boxes_with_label_and_scores(rgb_img, rbboxes, rclasses, rscores)
                final_detections_img = cv.cvtColor(final_detections_img, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(save_path, img_name), final_detections_img)
                # resize boxes and image according to raw input image
                # final_detections= cv.resize(final_detections[:, :, ::-1], (raw_w, raw_h))

                # recover to raw size
                detect_dict['score'] = rscores
                detect_dict['boxes'] = rbboxes
                detect_dict['categories'] = rclasses
                # convert from RGB to BG
                fw.write(f'\n{img_name}')
                for score, boxes, categories in zip(rscores, rbboxes, rclasses):
                    fw.write('\n\tscore:' + str(score))
                    fw.write('\tbboxes:' + str(boxes))
                    fw.write('\tcategories:' + str(int(categories)))

                view_bar('{} image cost {} second'.format(img_name, (end_time - start_time)), index + 1,
                               len(image_name_list))

            fw.close()


    def image_process(self, image, img_shape=(300, 300), img_format='NHWC'):
        data_format = 'NHWC'
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre = ssd_preprocessing.preprocess_for_eval(
            image, None, None, img_shape, img_format, resize=cfgs.Resize.WARP_RESIZE)

        return image_pre, labels_pre, bboxes_pre


if __name__ == "__main__":


    ckpt_path = '/home/alex/Documents/pretrain_model/ssd/ssd_300/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

    ssd_inference = ObjectInference(net_shape=(300, 300), ckpt_path=ckpt_path)

    ssd_inference.exucute_detect(image_path='./demo', save_path=cfgs.INFERENCE_SAVE_PATH)