import tensorflow

from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet',include_top=False)
print(model)