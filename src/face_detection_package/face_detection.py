from __future__ import division


import dlib
import cv2

import rospy


class landmarks_extractor:

	def __init__(self):
		#Set up some required objects
		self.detector = dlib.get_frontal_face_detector() #Face detector
		self.predictor = dlib.shape_predictor("/home/zoid/catkin_ws/src/extract_face_feature/dat/shape_predictor_68_face_landmarks.dat") #Landmark identifier.


	def prepareImage (self, frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		self.frame = frame
		self.preparedFrame = clahe.apply(gray)


	def getLandmarksImage (self):
		for index in range(len(self.faces)):
			landmarksImage = self.getLandmarksImageForFace(index)
			
		return landmarksImage


	def getLandmarksImageForFace (self, faceId):
		landmarksImage = self.frame

		for landmark in self.faces[faceId].landmarks:
			cv2.circle(landmarksImage, (landmark.x, landmark.y), 1, (0,0,255), thickness=2) 

		return landmarksImage


	def getNormalizedLandmarksImage (self):
		for index in range(len(self.faces)):
			landmarksImage = self.getNormalizedLandmarksImageForFace(index)
			
		return landmarksImage


	def getNormalizedLandmarksImageForFace (self, faceId):
		face = self.faces[faceId]

		face.setScaleTargetSize(256)	
	
		rospy.logdebug("Image Size %sx%s", int(256*face.horizontalRatio), int(256*face.verticalRatio))
		#crop image [Y1:Y2, X1:X2] 
		landmarksImage = cv2.resize(self.frame[face.minPoint.y:face.maxPoint.y, face.minPoint.x:face.maxPoint.x], (int(256*face.horizontalRatio), int(256*face.verticalRatio))) 

		for landmark in face.scaledLandmarks:
			cv2.circle(landmarksImage, (landmark.x, landmark.y), 1, (0,0,255), thickness=2) 

		return landmarksImage
		


	def extract (self):
		detections = self.detector(self.preparedFrame, 1) #Detect the faces in the image

		count = 0
		self.faces = []
		for k,d in enumerate(detections): #For each detected face
			self.faces.append(landmarks_face(self.predictor(self.preparedFrame, d), count))
			count += 1
		
		return self.faces
		


class landmarks_face:

	def __init__(self, shape, id):
	
		self.minPoint = point(-1, -1)
		self.maxPoint = point(-1, -1)
		
		self.landmarks = []

		self.verticalRatio = 1
		self.horizontalRatio = 1		
		
		for i in range(1,68): #There are 68 landmark points on each face
			self.landmarks.append(point(shape.part(i).x, shape.part(i).y))
			
			#determine min and max x & y
			if self.minPoint.x == -1 or self.minPoint.x > shape.part(i).x:
				self.minPoint.x = self.landmarks[i-1].x
		
			if self.minPoint.y == -1 or self.minPoint.y > shape.part(i).y:
				self.minPoint.y = self.landmarks[i-1].y

			if self.maxPoint.x == -1 or self.maxPoint.x < shape.part(i).x:
				self.maxPoint.x = self.landmarks[i-1].x
		
			if self.maxPoint.y == -1 or self.maxPoint.y < shape.part(i).y:
				self.maxPoint.y = self.landmarks[i-1].y			
			
	def getLandmarks (self):
		return self.landmarks
			
			
	def setScaleTargetSize(self, targetSize):
		self.scaleTargetSize = targetSize
		
		self.verticalRatio = 1
		self.horizontalRatio = 1
		width = self.maxPoint.x-self.minPoint.x
		height = self.maxPoint.y-self.minPoint.y
		
		if width > height:
			self.verticalRatio = height/width
		else:
			self.horizontalRatio = width/height
		
		scaleFactorX = targetSize/width*self.horizontalRatio
		scaleFactorY = targetSize/height*self.verticalRatio

		rospy.logdebug("Width/Height %s/%s", width, height)
		rospy.logdebug("Ratio H/V %s/%s", self.horizontalRatio, self.verticalRatio)		
		rospy.logdebug("Scale ratio X/Y %s/%s", scaleFactorX, scaleFactorY)

		self.setScaleFactor(scaleFactorX, scaleFactorY)

		
	def getVerticalRatio (self):
		return self.verticalRatio

		
	def getHorizontalRatio (self):
		return self.horizontalRatio		
		

	def setScaleFactor(self, scaleFactorX, scaleFactorY):

		self.scaledLandmarks = []
		for landmark in self.landmarks:
			self.scaledLandmarks.append(point(int((landmark.x-self.minPoint.x)*scaleFactorX), int((landmark.y-self.minPoint.y)*scaleFactorY)))
			
			
	def getScaledLandmarks (self):
		return self.scaledLandmarks
					
			
			
			
class point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def getX ():
		return x
		
	def getY ():
		return y
