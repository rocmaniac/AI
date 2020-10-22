import cv2
import time
import imutils
cam=cv2.VideoCapture(0)
time.sleep(2)
countFrame=None
area_within=500
objectCount=0
while True:
    _,img=cam.read()
    display="NOrmal"
    img= imutils.resize(img,width=500)
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussimg=cv2.GaussianBlur(grayimg,(21,21),0)
    if countFrame is None:
        countFrame=gaussimg
        continue
    imgDifference=cv2.absdiff(countFrame,gaussimg)
    thresimg=cv2.threshold(imgDifference,25,255,cv2.THRESH_BINARY)[1]
    thresDil=cv2.dilate(thresimg,None, iterations=2)
    cnts=cv2.findContours(thresDil.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for area in cnts:
        if cv2.contourArea(area)<area_within:
            continue
        objectCount+=1
        (x,y,w,h)=cv2.boundingRect(area)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
        display="Object Detected Moving > " + str(objectCount)
    print(display)
    cv2.putText(img,display,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("cameraFeed",img)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



        
