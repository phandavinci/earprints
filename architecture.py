from PIL import ImageFont
from tensorflow.keras import layers
import tensorflow as tf
from collections import defaultdict
import visualkeras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, DepthwiseConv2D
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'blue'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'
color_map[DepthwiseConv2D]['fill'] = 'red'
font = ImageFont.truetype('/home/phanda/Downloads/Arial.ttf',size=100)
model = base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',  # Load pre-trained weights
        include_top=False,
        input_shape=(224, 224, 3),
        classes = 4
    )

visualkeras.layered_view(model, legend=True, font=font, color_map=color_map, to_file='arch.png').show()
