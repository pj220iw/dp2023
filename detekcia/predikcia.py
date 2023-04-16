from configparser import Interpolation
from typing_extensions import assert_never
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from training import make_model_keras
from tensorflow.keras.models import load_model
#from tensorflow.python.platform import gfile
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import glob

from plotdata import plot_confusion_matrix
from config import Config_classification
from config import new_size
from segmentation import segmentation_keras_load
from plotdata import plot_segmentation_test

batch_size = Config_classification.get('batch_size')
image_size = (new_size.get('width'), new_size.get('height'))
epochs = Config_classification.get('Epochs')
#path_to_pb = "D:\diplomovka\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Output\Models\model_fire_resnet_weighted_40_no_metric_simple/saved_model.pb"

def predikcia():
    #reconstructed_model = tf.keras.models.load_model("Output/Peter_extra/model_fire_resnet_weighted_40_no_metric_simple")
    #reconstructed_model = tf.keras.models.load_model("training/peter_8/save_at_10.h5") 
    reconstructed_model = tf.keras.models.load_model("training/peter_extra/save_at_10.h5")
    count = 1450
    while count < 1590:
    #-------Skuska Ohnom---------#
    #img = keras.preprocessing.image.load_img(
    #    "skuska_ohnom/neohen.jpg", target_size=image_size)
    #-------Tello edu video na obrázky------------#
        img = keras.preprocessing.image.load_img(
            "tello/video_to_frames/12_04_2023/zahrada_ohen_vecer_%d.jpg" % count, target_size=image_size)
    #---------Tello obrázky--------------#
    #img = keras.preprocessing.image.load_img(
    #    "tello/fotky/zahrada_4_resized.jpg", target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = reconstructed_model.predict(img_array)
        score = predictions[0]
        plt.plot(count, 100 * (1 - score), 'o')
        print("Score: %.2f" % score)
        print("This image is %.2f percent Fire and %.2f percent No Fire." % (100 * (1 - score), 100 * score))
        count += 1
    plt.ylim(0, 100)
    plt.ylabel('% výskytu ohňa')
    plt.xlabel('počet vyhodnotených fotiek')
    plt.show()
    