#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

# create cascade classifier

face_cascade=cv2.CascadeClassifier("C:\\Users\\Legion\\Downloads\\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("C:\\Users\\Legion\\Downloads\\haarcascade_eye.xml")
# to capture image from your webcam
video=cv2.VideoCapture(0)
a=0
while True:
    a=a+1
    check,frame=video.read()
# to convert image in gray format   
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#search for coordinates

    faces= face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors = 5)
    for x,y,w,h in faces:
        img =  cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # x,y are coordinates of your face and h,w are hieght and width
        roi_gray=gray_img[y:y+h,x:x+w]
    
        roi_color=img[y:y+h,x:x+w]
    
        eyes= eye_cascade.detectMultiScale(roi_gray,scaleFactor = 1.3,minNeighbors = 5)
    
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
  # to show image  
    cv2.imshow('imgage',frame)
    k=cv2.waitKey(1)
    # to stop showing image
    if k==ord("q"):
        break
video.release()
cv2.destroyAllWindows()   

