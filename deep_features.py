import argparse
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA

from os.path import basename, dirname

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import json
import requests


def fetch_modules():
    payload = "[\"cma:po\",\"\",\"\",100,true,[[\"module-type\",[\"image-feature-vector\"]]],null,[],null,[]]"

    response = requests.request("POST", "https://tfhub.dev/s/list", data=payload,
                                headers={'content-type': "application/json"})
    response = response.text
    response = response[response.find('\n'):]

    def parse_modules(reponse):
        modules = {}
        for x in json.loads(reponse)[0][2]:
            modules[x[2]] = 'https://tfhub.dev/google/%s/%s' % (x[2], x[11])
        return dict(filter(lambda x: 'tf2' not in x[0] and 'quantops' not in x[0], modules.items()))

    return parse_modules(response)


modules = {
    # 'mobilenet_v1_025_128': 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/3',
    # 'mobilenet_v1_050_128': 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/3',
    # 'mobilenet_v1_075_128': 'https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/3',
    # 'mobilenet_v1_100_128': 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/3',
    # 'mobilenet_v1_025_160': 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/3',
    # 'mobilenet_v1_050_160': 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/3',
    # 'mobilenet_v1_075_160': 'https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/3',
    # 'mobilenet_v1_100_160': 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/3',
    # 'mobilenet_v1_025_192': 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/3',
    # 'mobilenet_v1_050_192': 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/3',
    # 'mobilenet_v1_075_192': 'https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/3',
    # 'mobilenet_v1_100_192': 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/3',
    'mobilenet_v1_025_224': 'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/3',
    'mobilenet_v1_050_224': 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/3',
    'mobilenet_v1_075_224': 'https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/3',
    'mobilenet_v1_100_224': 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/3',
    # 'mobilenet_v2_035_96': 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/3',
    # 'mobilenet_v2_050_96': 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/3',
    # 'mobilenet_v2_075_96': 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/3',
    # 'mobilenet_v2_100_96': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/3',
    # 'mobilenet_v2_035_128': 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/3',
    # 'mobilenet_v2_050_128': 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/3',
    # 'mobilenet_v2_075_128': 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/feature_vector/3',
    # 'mobilenet_v2_100_128': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/3',
    # 'mobilenet_v2_035_160': 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/feature_vector/3',
    # 'mobilenet_v2_050_160': 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/3',
    # 'mobilenet_v2_075_160': 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/feature_vector/3',
    # 'mobilenet_v2_100_160': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/3',
    # 'mobilenet_v2_035_192': 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/feature_vector/3',
    # 'mobilenet_v2_050_192': 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/3',
    # 'mobilenet_v2_075_192': 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/feature_vector/3',
    # 'mobilenet_v2_100_192': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/3',
    'mobilenet_v2_035_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/3',
    'mobilenet_v2_050_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/3',
    'mobilenet_v2_075_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/3',
    'mobilenet_v2_100_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3',
    'mobilenet_v2_130_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/3',
    'mobilenet_v2_140_224': 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3',
    'resnet_v1_50': 'https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/3',
    'resnet_v1_101': 'https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/3',
    'resnet_v1_152': 'https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/3',
    'resnet_v2_50': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3',
    'resnet_v2_101': 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/3',
    'resnet_v2_152': 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3',
    'inception_v1': 'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/3',
    'inception_v2': 'https://tfhub.dev/google/imagenet/inception_v2/feature_vector/3',
    'inception_v3': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3',
    'inception_resnet_v2': 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/3',
    'nasnet_mobile': 'https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/3',
    'nasnet_large': 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/3',
    'pnasnet_large': 'https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/3',
    'amoebanet_a_n18_f448': 'https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/feature_vector/1',
    'efficientnet_b0': 'https://tfhub.dev/google/efficientnet/b0/feature-vector/1',
    'efficientnet_b1': 'https://tfhub.dev/google/efficientnet/b1/feature-vector/1',
    'efficientnet_b2': 'https://tfhub.dev/google/efficientnet/b2/feature-vector/1',
    'efficientnet_b3': 'https://tfhub.dev/google/efficientnet/b3/feature-vector/1',
    'efficientnet_b4': 'https://tfhub.dev/google/efficientnet/b4/feature-vector/1',
    'efficientnet_b5': 'https://tfhub.dev/google/efficientnet/b5/feature-vector/1',
    'efficientnet_b6': 'https://tfhub.dev/google/efficientnet/b6/feature-vector/1',
    'efficientnet_b7': 'https://tfhub.dev/google/efficientnet/b7/feature-vector/1',
}


def _parse_images(path, label, size):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])
    return image, label, path


def input_fn(paths, labels, size, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda x, y: _parse_images(x, y, size))
    # dataset = dataset.shuffle(buffer_size=len(labels))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    images_batch, labels_batch, filenames_batch = iterator.get_next()
    return images_batch, labels_batch, filenames_batch


def run(dataset_folder, network='inception_v3', batch_size=16):
    assert network in modules, 'Invalid network, pick one of %s' % list(modules.keys())
    assert dataset_folder is not None

    with tf.Graph().as_default():
        dataset = basename(dirname(dataset_folder))
        filenames_output = os.path.join(dataset_folder, dataset + '_' + network + '_filenames.csv')
        labels_output = os.path.join(dataset_folder, dataset + '_' + network + '_labels.csv')
        features_output = os.path.join(dataset_folder, dataset + '_' + network + '_features.csv')

        module_url = modules[network]

        types = ('/*/*.jpg', '/*/*.png')
        filenames = []
        for files in types:
            filenames.extend(glob.glob(dataset_folder + files))
        pbar = tqdm(total=len(filenames))
        labels = [basename(dirname(f)) for f in filenames]
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)

        module_spec = hub.load_module_spec(module_url)
        output_size = module_spec.get_output_info_dict()['default'].get_shape()[1]
        height, width = hub.get_expected_image_size(module_spec)

        images, labels, files = input_fn(filenames, labels, [height, width], batch_size)

        features = np.empty((0, output_size), float)
        classes = np.empty(0, int)
        filenames = np.empty(0, str)

        network = hub.Module(module_spec)
        network = network(images), labels, files

        with tf.compat.v1.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())

            while True:
                try:
                    x, y, f = sess.run(network)
                    f = [basename(k) for k in f]
                    filenames = np.append(filenames, f)
                    classes = np.append(classes, y)
                    features = np.append(features, x, axis=0)
                    pbar.update(len(y))
                except tf.errors.OutOfRangeError:
                    break
        pbar.close()

        # pca = PCA(n_components=100, random_state=1)
        # features = pca.fit_transform(features)
        np.savetxt(filenames_output, filenames.astype(str), fmt='%s', delimiter=',')
        np.savetxt(labels_output, classes.astype(str), fmt='%s', delimiter=',')
        np.savetxt(features_output, features, delimiter=',')


if __name__ == '__main__':
    os.path.join('')
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_folder', help="Folder containing image dataset")
    parser.add_argument('-network', default='inception_v3', help='default: \'inception_v3\'; Networks avaliable: %s' % list(modules.keys()))
    parser.add_argument('-batch_size', default=16, type=int, help="default: 16")
    args = parser.parse_args()

    run(dataset_folder=args.dataset_folder, network=args.network, batch_size=args.batch_size)

