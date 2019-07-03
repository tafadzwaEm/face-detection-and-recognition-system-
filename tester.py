import os 
import cv2 
import numpy as np 
import faceRecognition as fr 

cap = cv2.VideoCapture(0)
faces,faceId = fr.labelsForTrainingData('trainingImages')
faceRecognizer = fr.trainer(faces,faceId)
names = {0:'Shingi',1:'Emmanuel',2:"Flossy",4:"peter-dinklage",5:"kit-harington"}

while True:
    ret,testImage = cap.read()
    facesDetected,testImageGray = fr.faceDetection(testImage)

    for face in facesDetected:
        x,y,w,h = face
        roi = testImageGray[y:y+h,x:x+w]
        label,confidence = faceRecognizer.predict(roi)
        print(f'confidence: {confidence}')
        print(f'name: {label}')
        predictedName = names[label]
        cv2.rectangle(testImage,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(testImage,predictedName,(x,y),cv2.FONT_HERSHEY_DUPLEX,1.3,(0,0,255),2)
        
    resizedImage = cv2.resize(testImage,(500,400))
    cv2.imshow('faces',resizedImage)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  