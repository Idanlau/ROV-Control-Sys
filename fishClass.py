import cv2
import tensorflow as tf

from tensorflow import keras

import numpy as np

import time

model = tf.keras.models.load_model(
    "/Users/idanlau/Desktop/model", custom_objects=None, compile=True, options=None
)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/Users/idanlau/Desktop/ROV Image Detection/Train",
  seed=123,
  labels='inferred',
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)




cap = cv2.VideoCapture(0)

time.sleep(3)

ret,frame = cap.read()


img = cv2.resize(frame, (img_width,img_height),interpolation=cv2.INTER_AREA)
cv2.imwrite('/Users/idanlau/Desktop/image.jpg',frame)

# img = tf.keras.utils.load_img(
#     frame, target_size=(img_height, img_width))

cv2.imshow('',img)
cv2.waitKey(0)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()




