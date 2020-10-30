import cv2, os, urllib, numpy, imutils
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'  
sub_data = 'kohli'     
url="http://192.168.1.2:8080/shot.jpg"
path = os.path.join(datasets, sub_data)  #create dataset dir first 
#path=path.replace('\\','/')
print(path)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)
#webcam = cv2.VideoCapture(0)

count = 1
while count < 50:
    print(count)
    #(_, im) = webcam.read()
    imgPath=urllib.request.urlopen(url)
    imgNp=numpy.array(bytearray(imgPath.read()),
                      dtype=numpy.uint8)
    img=cv2.imdecode(imgNp,-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)
    count += 1
	
    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == 27:
        break
print("Dataset obtained successfully")
#webcam.release()
cv2.destroyAllWindows()
