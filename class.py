
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
import numpy as np

from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K



IM_WIDTH, IM_HEIGHT = 224,224
train_dir = '/home/liuyp/liu/keras_mobilenet/data3/train'
val_dir = '/home/liuyp/liu/keras_mobilenet/data3/validation'


nb_classes= 6
nb_epoch = 20

batch_size = 32
nb_train_samples=2327
nb_val_samples=2479

train_datagen =ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
test_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(IM_WIDTH, IM_HEIGHT),batch_size=batch_size,class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(val_dir,target_size=(IM_WIDTH, IM_HEIGHT),batch_size=batch_size,class_mode='categorical')


base_model=MobileNet(include_top=False, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Reshape((1,1,1024), name='reshape_1')(x)
x = Dropout(1e-3, name='dropout')(x)
x = Conv2D(6, (1, 1),padding='same', name='conv_preds')(x)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((6,), name='reshape_2')(x)

model = Model(inputs=base_model.input,outputs=x)
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
history_tl = model.fit_generator(train_generator,nb_epoch=nb_epoch,samples_per_epoch=nb_train_samples/16,validation_data=validation_generator,nb_val_samples=nb_val_samples/16,class_weight='auto')
model.save("model123.h5")