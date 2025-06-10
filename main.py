from utils.utils import brain
import cv2
import tensorflow as tf
import numpy as np

from utils.argparser import get_parser
from src.control import MovementControl
from src.control_test import TestMovementControl, TestMovementControl_1

import os
import asyncio

model_path = "/home/aiwearable/EmotionDetection-Wearable/models/model_optimized.tflite"
casc_path  = "/home/aiwearable/EmotionDetection-Wearable/haarcascade_frontalface_default.xml"
camera_path = "/dev/my_cam"

emotion = "neutual"  # Default emotion

async def detection_loop(args):
    global emotion

    args = get_parser()

    # Connecting the movement control
    # control = MovementControl(args)

    print('Loading the model...')

    f = tf.lite.Interpreter(model_path)
    f.allocate_tensors()
    i = f.get_input_details()[0]
    o = f.get_output_details()[0]

    print('Loading Successful !')

    faceCascade = cv2.CascadeClassifier(casc_path)

    cap = cv2.VideoCapture(camera_path)
    ai = 'anger'
    img = np.zeros((200, 200, 3))
    ct = 0
    try:
        while(True):

            # print('Capturing frame...')
            
            ret, frame = cap.read()

            # Retry logic for reading frames
            if not ret:
                print(f"Error reading frame, trying to reconnect...")

                if cap.isOpened():
                    cap.release()

                if os.path.exists(camera_path):
                    # If the camera path exists, try to reconnect
                    cap = cv2.VideoCapture(camera_path)

                asyncio.sleep(1)  # Wait for a second before retrying
                continue
        
            ct+=1

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(150, 150)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, ai, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                ## Change here for time between updates
                #if ct > 3 by default
                if ct > args["update_interval"]:
                    ai = brain(gray, x, y, w, h, f, i, o)
                    ct = 0

            emotion = ai
            await asyncio.sleep(5 / 1000)

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Detection loop interrupted by user.")
        cap.release()
        cv2.destroyAllWindows()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

async def control_loop(args):
    global emotion

    # Connecting the movement control
    control = MovementControl(args)

    while True:
        if emotion == "sadness":
            print("Sadness detected, moving servo...")
            pass

        elif emotion == "anger":
            print("Anger detected, moving servo...")
            pass

        elif emotion == "happy":
            print("Happiness detected, moving servo...")
            pass

        elif emotion == "neutral":
            print("Neutral emotion detected, no movement.")
            pass

        await asyncio.sleep(1000/1000)  # Control interval

async def main(args):
    await asyncio.gather(
        detection_loop(args),
        control_loop(args)
    )

if __name__ == '__main__':
    args = get_parser()
    asyncio.run(main(args))


