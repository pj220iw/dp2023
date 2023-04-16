from configparser import Interpolation
from typing_extensions import assert_never
from tensorflow import keras
from training import make_model_keras
#from tensorflow.python.platform import gfile
from config import Config_classification
from config import new_size

image_size = (new_size.get('width'), new_size.get('height'))

def resize():

    img = keras.preprocessing.image.load_img(
        "tello/fotky/zahrada_4.png", target_size=image_size)

    resized_img = img.resize((512,512))
    resized_img.save('tello/fotky/zahrada_4_resized.jpg')