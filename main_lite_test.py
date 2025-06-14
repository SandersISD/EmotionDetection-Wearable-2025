from utils.utils import brain
import cv2
import tensorflow as tf
import numpy as np

from utils.argparser import get_parser
from src.control import MovementControl
from src.control_test import TestMovementControl, TestMovementControl_1


if __name__ == '__main__':

    args = get_parser()

    # Connecting the movement control
    #test_control = TestMovementControl_1(args)

    control = MovementControl(args)

    print('Loading the model...')

    f = tf.lite.Interpreter("/home/aiwearable/EmotionDetection-Wearable/models/model_optimized.tflite")
    f.allocate_tensors()
    i = f.get_input_details()[0]
    o = f.get_output_details()[0]

    print('Loading Successful !')

    cascPath = "/home/aiwearable/EmotionDetection-Wearable/haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    # cap = cv2.VideoCapture(-1)
    cap = cv2.VideoCapture("/dev/video0")
    ai = 'anger'
    img = np.zeros((200, 200, 3))
    ct = 0
    while(True):

        print('Capturing frame...')
        
        ret, frame = cap.read()
        
        # if not frame:
        #     print(f"Error reading frame:")
            # cap.release()
            # i %= 2
            # cap = cv2.VideoCapture("/dev/video{i}")
    
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
        
        ano = ''    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, ai, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            ## Change here for time between updates
            #if ct > 3 by default
            if ct > args["update_interval"]:
                
                #Connecting to servo and everything

                ai = brain(gray, x, y, w, h, f, i, o)
                ## Mechanical move here 
                # if ai == "sadness":
                    #test_control.Sadness()
                    # control.Sadness()

                # elif ai == "anger":
                    #test_control.Anger()
                    # control.Anger()

                # elif ai == "happy":
                    # test_control.Happy()
                    # control.Happy()
                    
                ct = 0

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()