import os 
import cv2 
import numpy as np 

def faceDetection(testImage):
    testImageGray = cv2.cvtColor(testImage,cv2.COLOR_BGR2GRAY)
    faceCascades = cv2.CascadeClassifier('haarCascades/haarcascade_frontalface_default.xml')
    facesDetected = faceCascades.detectMultiScale(testImageGray,scaleFactor=1.5,minNeighbors=5)
    return facesDetected,testImageGray 

def labelsForTrainingData(directory):
    faces = []
    faceId = []
    for root,dirs,files in os.walk(directory):
        for fil in files:
            if fil.startswith('.'):
                continue
            imagePath = os.path.join(root,fil)
            imageId = os.path.basename(root)
            testImage = cv2.imread(imagePath)
            if testImage is None:
                print('image did not load properly')
                continue
            facesDetected,testImageGray = faceDetection(testImage)
            if len(facesDetected) !=1:
                print('skipping images with more than one face')
                continue
            x,y,w,h = facesDetected[0]
            roiGray = testImageGray[y:y+h,x:x+w]
            faces.append(roiGray)
            faceId.append(int(imageId))
    return faces,faceId

def trainer(faces,faceId):
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.train(faces,np.array(faceId))
    return faceRecognizer










    
