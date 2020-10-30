import cv2
import numpy
import os
import urllib.request
haar_file='haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)
datasets='dataset'
print('traning...')
url="http://192.168.1.2:8080/shot.jpg"
images,labels,names,id=[],[],{},0
count=1
for subdirs, dirs, files in os.walk(datasets):
    count+=1
    for subdir in dirs:
        names[id]=subdir
        subjectpath =os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path= subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
images, labels = [numpy.array(lis) for lis in [images, labels]]
width, height= 130,100
model=cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
cnt=0
while True:
    imgPath=urllib.request.urlopen(url)
    imgNp=numpy.array(bytearray(imgPath.read()),
                      dtype=numpy.uint8)
    im=cv2.imdecode(imgNp,-1)
    gray =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h), (255,255,255),4)
        face= gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction =model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,255),4)
        if prediction[1]<800:
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 255))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im,'unknowN',
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255))
            print("Unknown")
            cv2.write("unknown.jpg",im)
            cnt=0
    
    cv2.imshow("FaceRecongintion",im)
    if cv2.waitKey(10)==27:
        break
cv2.destroyAllWindows()













