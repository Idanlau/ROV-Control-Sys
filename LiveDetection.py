import cv2
import tensorflow as tf

from tensorflow import keras

import numpy as np

import time

def run():
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
    kernel = np.ones((2, 2), np.uint8)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        org = (50, 50)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.medianBlur(gray, 3)  # to remove salt and paper noise
        # to binary
        ret,thresh = cv2.threshold(gray,129,129,129) # to detect gray objects

        # to get outer boundery only
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        # to strength week pixels
        thresh = cv2.dilate(thresh, kernel, iterations=5)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image = frame

        if len(contours) > 0:
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), 5)
            # find the biggest countour (c) by the area
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # draw the biggest contour (c) in green
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            image = frame[y:y+h,x:x+w]
            # Display the resulting frame
            # print(len(image))
            # print(len(frame))

            #cv2.imshow('frame', image)


            img = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])



            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2

            font = cv2.FONT_HERSHEY_SIMPLEX

            frame = cv2.putText(frame, str(class_names[np.argmax(score)])+": "+str(100 * np.max(score)), org, font,
                           fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('frame', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('/Users/idanlau/Desktop/image.jpg', img)
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
