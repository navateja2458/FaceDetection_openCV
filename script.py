import cv2 as c

face_cascade=c.CascadeClassifier("haarcascade_frontalface_default.xml")
img=c.imread("photo.jpg")
gray_img=c.cvtColor(img,c.COLOR_BGR2GRAY)

#c.imshow("gray imge",gray_img)

faces=face_cascade.detectMultiScale(gray_img,minNeighbors=5,scaleFactor=1.2)

for x,y,w,d in faces:
   img=c.rectangle(img,(x,y),(x+w,y+d),(255,126,0),3)

resized_img=c.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

c.imshow("detect_face",resized_img)
c.waitKey(0)
c.destroyAllWindows()


