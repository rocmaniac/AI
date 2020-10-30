from facial_emotion_recognition import EmotionRecognition
import cv2
er=EmotionRecognition(device='gpu') #device='cpu', change if you dont have GPU
cam =cv2.VideoCapture(0)

while True:
    _,img =cam.read()
    img=er.recognise_emotion(img,return_type='BGR')
    cv2.imshow('frame',img)
    if cv2.waitKey(1)==27:#esc
        break
cam.release()
cv2.destroyAllWindows()
