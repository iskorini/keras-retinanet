#!/usr/bin/env python
# coding: utf-8

# In[1]:

import subprocess, re

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

#import setGPU

from comet_ml import Experiment
from comet_ml import Optimizer
import tensorflow as tf
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#import comet_ml

API_KEY = 'spsZWUefep11cUugBplS6AKQm' #os.environ['COMET_API_KEY']

# import keras
import keras
from keras import backend as K 

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.losses import smooth_l1, focal
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet.utils.eval import evaluate

# other imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# set tf backend to allow memory to grow, instead of claiming everything


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(get_session())

# In[2]:


default_params = {
    'N': 3,
    'M': 10
}
dataset_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/fine_tuning_kaist_cars/ds')
train_annotations = dataset_path.joinpath('train_no_people.csv')
test_annotations = dataset_path.joinpath('test_w_people.csv')
classes = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/KAIST_MPD/class_name_to_ID_CARS.csv')

def create_train_generator(N, M):
    train_generator = csv_generator.CSVGenerator(
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

def create_test_generator():
    test_generator = csv_generator.CSVGenerator(
    test_annotations,
    classes,
    image_min_side=800,
    image_max_side=1333,
    config=None,
    shuffle_groups=False)
    return test_generator
    
#train_generator = create_train_generator(default_params['N'], default_params['M'])
#test_generator = create_test_generator()


# In[3]:


def create_model(train_generator): 
    model_path = Path('/data/students_home/fschipani/thesis/MSc-Thesis-PJ/Dataset/weights/autoaugment/v2/resnet50_csv_05.h5')
    model = models.load_model(model_path, backbone_name='resnet50')
    model.compile(
        loss={
            'regression'    : smooth_l1(),
            'classification': focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    return model


# In[4]:


def test_model(model, _generator, score_threshold):
    model = models.convert_model(model, anchor_params=None) 
    result = evaluate(
        _generator,
        model,
        score_threshold=0.3,
        iou_threshold=0.5,
        max_detections=100,
        save_path=None
    )
    return result

#test_results = test_model(create_model(), create_test_generator(), 0.3)


# In[5]:


#test_results[2][0] #mAP for person


# In[6]:


config = {
      "algorithm": "bayes",
      "parameters": {
          "N": {"type": "integer", "min": 1, "max": 3},
          "M": {"type": "integer", "min": 5, "max": 30},
      },
      "spec": {
          "metric": "mAP",
          "objective": "minimize",
      },
  }
#train_generator = create_train_generator(3, 30)
#model = create_model(train_generator)
#test_results = test_model(model, create_test_generator(), 0.3)
#print(test_results)
#for label, (average_precision, num_annotations) in test_results.items():
#            print('{:.0f} instances of class'.format(num_annotations),
#                  train_generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
# In[ ]:

opt = Optimizer(config, api_key=API_KEY, project_name="MSC-Thesis-PJ-PERSON-AUTOAUGMENT") #SOSTITUIRE CON CARS!
for experiment in opt.get_experiments():
    N = experiment.get_parameter('N')
    M = experiment.get_parameter('M')
    train_generator = create_train_generator(N, M)
    model = create_model(train_generator)
    history = model.fit_generator(generator=train_generator, steps_per_epoch=1750, epochs=5, verbose=1, #sostituire con 1750!!!
        validation_data = create_test_generator(), validation_steps = 500, validation_freq = 1)
    test_results = test_model(model, create_test_generator(), 0.3)
    print(test_results)
    experiment.log_metric("mAP", -test_results[0][0])
    K.clear_session()
    
