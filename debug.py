#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import cv2

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.kitti import KittiGenerator
from keras_retinanet.preprocessing.open_images import OpenImagesGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.visualization import draw_annotations, draw_boxes, draw_caption
from keras_retinanet.utils.anchors import anchors_for_shape, compute_gt_annotations
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.image import random_visual_effect_generator
from pathlib import Path


def create_generator(N,M):
    dataset_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/fine_tuning_kaist_cars/ds')
    train_annotations = dataset_path.joinpath('train_no_people.csv')
    classes = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/class_name_to_ID_CARS.csv')
    train_generator = CSVGenerator(
        train_annotations,
        classes,
        transform_generator=None,
        visual_effect_generator=None,
        image_min_side=800,
        image_max_side=1333,
        auto_augment=None,
        rand_augment=(N,M),
        config=None
        )
    return train_generator



def make_output_path(output_dir, image_path, flatten = False):
    """ Compute the output path for a debug image. """

    # If the output hierarchy is flattened to a single folder, throw away all leading folders.
    if flatten:
        path = os.path.basename(image_path)

    # Otherwise, make sure absolute paths are taken relative to the filesystem root.
    else:
        # Make sure to drop drive letters on Windows, otherwise relpath wil fail.
        _, path = os.path.splitdrive(image_path)
        if os.path.isabs(path):
            path = os.path.relpath(path, '/')

    # In all cases, append "_debug" to the filename, before the extension.
    base, extension = os.path.splitext(path)
    path = base + "_debug" + extension

    # Finally, join the whole thing to the output directory.
    return os.path.join(output_dir, path)


def run(generator):
    """ Main loop.

    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    i = 0
    while True:
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)
        if len(annotations['labels']) > 0 :
            print(generator.image_path(i))
            anchors = anchors_for_shape(image.shape, anchor_params=None)
            positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])
            image, annotations = generator.rand_augment_group_entry(image, annotations)
            if len(annotations['labels']>0):
                positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])
                draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)
                draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))
                draw_caption(image, [0, image.shape[0]], os.path.basename(generator.image_path(i)))
        if True:
            output_path = make_output_path('./debug', generator.image_path(i), flatten=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            i += 1
            if i == generator.size():  # have written all images
                break
            else:
                continue
    return True


def main(args=None):
    # create the generator
    while True:
        print('---------------------------------------------------------')
        print('---------------------------------------------------------')
        print('---------------------------------------------------------')
        print('---------------------------------------------------------')
        generator = create_generator(5, 10)
        run(generator)


if __name__ == '__main__':
    main()
