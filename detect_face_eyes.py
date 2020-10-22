import cv2
algo_face='haarcascade_frontalface_default.xml'
algo_eye='haarcascade_eye.xml'
haar_cascade=cv2.CascadeClassifier(algo_face)
eye_cascade=cv2.CascadeClassifier(algo_eye)
cam=cv2.VideoCapture(0)
while True:
    _,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(gray,1.3,4)
    display="NOT SPOTTED"
    eyes= eye_cascade.detectMultiScale(gray,1.3,4)     
    for (x, y, w, h) in eyes: 
        cv2.rectangle(img, (x, y),  
                      (x + w, y + h), (255, 255, 255), 2)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        display="FACE SPOTTED"
    cv2.putText(img,display,(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
    cv2.imshow("FACE SCREEN",img)
    key=cv2.waitKey(10)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
        
