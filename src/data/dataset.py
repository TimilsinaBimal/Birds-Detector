import os
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, root_dir, image_size=(224, 224, 3), batch_size=24, buffer_size=512):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS = image_size
        self.BATCH_SIZE = batch_size
        self.ROOT_DIR = root_dir
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.BASE_IMAGE_DIR = os.path.join(
            self.ROOT_DIR, self.DATA_DIR, "images")
        self.BASE_ANNOS_DIR = os.path.join(
            self.ROOT_DIR, self.DATA_DIR, "annotations")
        self.TRAIN_DATA_LEN = 0
        self.TEST_DATA_LEN = 0
        self.VAL_DATA_LEN = 0

    @staticmethod
    def list_files(base_path):
        filepaths = []
        subdirs = os.listdir(base_path)
        for subdir in subdirs:
            subdir_path = os.path.join(base_path, subdir)
            files = os.listdir(subdir_path)
            for file_p in files:
                filepaths.append(os.path.join(base_path, subdir, file_p))
        return filepaths

    def prepare_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=self.CHANNELS, dtype=tf.float16)
        shape = img.numpy().shape
        img = tf.image.resize(img, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH])
        return img, shape

    @staticmethod
    def get_bbox(filename):
        mat_file = scipy.io.loadmat(filename)
        bbox = mat_file['bbox'][0][0]
        left = bbox[0][0][0]
        top = bbox[1][0][0]
        right = bbox[2][0][0]
        bottom = bbox[3][0][0]
        return [left, top, right, bottom]

    def split_data(self, images, bboxes):
        train_images, val_images, train_annos, val_annos = train_test_split(
            images, bboxes, test_size=0.15, random_state=42)
        train_images, test_images, train_annos, test_annos = train_test_split(
            train_images, train_annos, test_size=0.10, random_state=42)

        return train_images, val_images, test_images, train_annos, val_annos, test_annos

    def resize_bbox(self, bbox, shape):
        height, width, _ = shape
        left, top, right, bottom = bbox
        factor_x = self.IMAGE_WIDTH / width
        factor_y = self.IMAGE_HEIGHT / height
        left = left * factor_x
        top = top * factor_y
        right = right * factor_x
        bottom = bottom * factor_y
        return np.array([left, top, right, bottom])

    def prepare_data(self, images, annotations):
        for idx in range(len(images)):
            image = images[idx]
            annotation = annotations[idx]
            image, shape = self.prepare_image(image)
            bbox = self.get_bbox(annotation)
            bbox = self.resize_bbox(bbox, shape)
            yield image, bbox

    def prepare_dataset(self, images, annotations):
        dataset = tf.data.Dataset.from_generator(
            self.prepare_data,
            args=[images, annotations],
            output_types=(tf.float16, tf.float16),
            output_shapes=(
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS),
                (4,)
            )
        )
        return dataset

    def prepare(self):
        images = self.list_files(self.BASE_IMAGE_DIR)
        annos = self.list_files(self.BASE_ANNOS_DIR)
        train_images, val_images, test_images, train_annos, val_annos, test_annos = self.split_data(
            images, annos)

        self.TRAIN_DATA_LEN = len(train_images)
        self.VAL_DATA_LEN = len(val_images)
        self.TEST_DATA_LEN = len(test_images)

        train_dataset = self.prepare_data(train_images, train_annos)
        val_dataset = self.prepare_data(val_images, val_annos)
        test_dataset = self.prepare_data(test_images, test_annos)
        train_dataset = train_dataset.shuffle(
            512).repeat().batch(self.BATCH_SIZE)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = val_dataset.shuffle(200).batch(
            self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.shuffle(
            100).batch(1).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset
