# -*- coding:utf-8 -*-
#
# Created by lee
#
# 2018-04-15

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import yolo.config as cfg

class preprocess(object):
    def __init__(self, rebuild=False):
        self.data_path = os.path.join(cfg.DATA_PATH, 'data_set')
        self.output_path = os.path.join(cfg.DATA_PATH, 'output')
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.rebuild = rebuild
        self.cursor = 0
        self.cursor_test = 0
        self.epoch = 1
        self.gt_labels = None

    def next_batches(self, gt_labels, batch_size):
        images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((batch_size, self.cell_size, self.cell_size, self.num_class + 5))
        count = 0
        while count < batch_size:
            imname = gt_labels[self.cursor]['imname']
            flipped = gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(gt_labels):
                np.random.shuffle(gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def next_batches_test(self, gt_labels, batch_size):
        images = np.zeros((batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((batch_size, self.cell_size, self.cell_size, self.num_class + 5))

        count = 0
        while count < batch_size:
            imname = gt_labels[self.cursor_test]['imname']
            flipped = gt_labels[self.cursor_test]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = gt_labels[self.cursor_test]['label']
            count += 1
            self.cursor_test += 1
            if self.cursor_test >= len(gt_labels):
                np.random.shuffle(gt_labels)
                self.cursor_test = 0
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self, model):
        gt_labels = self.load_labels(model)
        '''if self.flipped:
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                print(gt_labels_cp[idx]['label'], '   ', gt_labels_cp[idx]['label'][:, ::-1, :])
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp'''
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self, model):
        if model == 'train':
            txtname = os.path.join(self.data_path, 'train.txt')
        if model == 'test':
            txtname = os.path.join(self.data_path, 'test.txt')

        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'Images', index + '.jpg')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        return gt_labels

    def load_pascal_annotation(self, index):
        imname = os.path.join(self.data_path, 'Images', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, self.num_class + 5))
        filename = os.path.join(self.data_path, 'Labels', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(bbox.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(bbox.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(bbox.find('ymax').text)) * h_ratio, self.image_size), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
