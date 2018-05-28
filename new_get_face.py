#!/usr/bin/python
import cv2
import numpy as np
import time
c = time.time()
t = time.ctime(c)
i=1
'''
videoname = str(t)+'.avi'
fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', '2')
out=cv2.VideoWriter(videoname,fourcc,10,(640,480))
'''
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:\\bishe\\faceProject\\images\dataset\\aa.avi', fourcc, 20.0, (604, 480))
cap = cv2.VideoCapture(0) #此处填入摄像头
while(i < 10):
    ret_flag, frame=cap.read()   #get frame
    cv2.imshow("takePhoto",frame)
    #
    if i :
        cv2.imwrite('D:\\bishe\\faceProject\\images\dataset\\'+str(i)+'.jpg', frame)
    i = i + 1
    #out.write(frame)
    if cv2.waitKey(1)&0xFF==ord('q')or ret_flag==False:
        break
cap.release()
cv2.destroyAllwindows()