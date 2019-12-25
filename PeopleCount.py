import cv2
import glob
import time
import numpy as np
import tensorflow.compat.v1 as tf
from imutils.video import VideoStream
from People_Counter import *


tf.disable_v2_behavior()

vs = VideoStream(src = 0).start()
time.sleep(2.0)

modelPath = 'Data/Model/my_model.pb'
PeopleConter = People_Counter(path = modelPath)
threshold = 0.4
no = 1

while True:
    count = 0
    frame = vs.read()
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640,480))
    cv2.imwrite("Data/Imagens/Frame.jpg",frame)
    img = cv2.imread("Data/Imagens/Frame.jpg")

    boxes, scores, classes, num = PeopleConter.detect(img)

    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i]>threshold:
            box = boxes[i]
            cv2.rectangle(img, (box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            count+=1
    
    print("Valor predito: {}".format(count))
    cv2.putText(img,'Count = '+str(count),(10,400),cv2.FONT_HERSHEY_SIMPLEX, 1.25,(255,255,0),2,cv2.LINE_AA)
    cv2.imwrite('Data/Resultados/resultado.jpg',img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    time.sleep(2.0)

