from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import pandas as pd
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
model = load_model("model.hdf5")

model_points = np.array([
							(0.0, 0.0, 0.0),             # Nose tip
							(0.0, -330.0, -65.0),        # Chin
							(-225.0, 170.0, -135.0),     # Left eye left corner
							(225.0, 170.0, -135.0),      # Right eye right corne
							(-150.0, -150.0, -125.0),    # Left Mouth corner
							(150.0, -150.0, -125.0)      # Right mouth corner
							
                        ])
def getYRP(image_points, size):
	center = (size[1]/2, size[0]/2)
	focal_length = center[0] / np.tan(60/2 * np.pi / 180)
	camera_matrix = np.array(
							[[focal_length, 0, center[0]],
							[0, focal_length, center[1]],
							[0, 0, 1]], dtype = "double"
							)
	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
	axis = np.float32([[500,0,0], 
							[0,500,0], 
							[0,0,500]])
							
	# imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	# modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
	proj_matrix = np.hstack((rvec_matrix, translation_vector))
	eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
	pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
	pitch = math.degrees(math.asin(math.sin(pitch)))
	roll = -math.degrees(math.asin(math.sin(roll)))
	yaw = math.degrees(math.asin(math.sin(yaw)))
	#print("YRP")
	#print(yaw, roll, pitch)
	return yaw/100, roll/100, pitch/100
	
def getArray(landmarks):
	image_points = np.array([
                            (landmarks[0][0], landmarks[0][1]),     # Nose tip
                            (landmarks[1][0], landmarks[1][1]),     # Chin
                            (landmarks[2][0], landmarks[2][1]),     # Left eye left corner
                            (landmarks[3][0], landmarks[3][1]),     # Right eye right corne
                            (landmarks[4][0], landmarks[4][1]),     # Left Mouth corner
                            (landmarks[5][0], landmarks[5][1])      # Right mouth corner
                        ], dtype="double")
	return image_points
def prepare_vector(ypr, feature_vec):
	n = len(ypr)
	if(n==0):
		return []
	single_vec = [0 for i in range(15)]
	for j in range(3):
		for i in range(n):
			single_vec[j] = single_vec[j] + abs(ypr[i][j])
	
	for i in range(6):
		for j in range(n):
			single_vec[i*2+3] = single_vec[i*2+3] + feature_vec[j][i][0]/270
			single_vec[i*2+4] = single_vec[i*2+4] + feature_vec[j][i][1]/270
	for i in range(15):
		single_vec[i] = single_vec[i]/n
	return single_vec

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def detectSmilesCNN(gray,rects):
	
	fX, fY, fW, fH = rect_to_bb(rects)
	# extract the ROI of the face from the grayscale image,
	# resize it to a fixed 28x28 pixels, and then prepare the
	# ROI for classification via CNN
	roi = gray[fY: fY + fH, fX: fX + fW]
	try:
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis = 0)

		# determine the probabilities of both "smiling" and "not similing"
		# then set the label accordingly
		(notSmiling, smiling) = model.predict(roi)[0]
		#print(notSmiling, smiling)
		label = "Smiling" if notSmiling < 0.6 else "Not Smiling"
		smiles.append(label)
	except Exception as e:
		print(e)

	#print(label)

def generateVideoFeatures(path):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	count = 0
	success = 1
	vidObj = cv2.VideoCapture(path) 
	feature_vec = []
	ypr =[]

	while count<75: 
		success, image = vidObj.read()
		if success:
			count+=1
			try:
				image = imutils.resize(image, width=500)
			except:
				continue
			size = image.shape
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# detect faces in the grayscale image
			rects = detector(gray, 1)
			# determine the facial landmarks for the face region, then
			# convert the landmark (x, y)-coordinates to a NumPy array
			if(len(rects) == 0):
				continue
			#detectSmiles(gray, image, rects[0])
			detectSmilesCNN(gray,rects[0])
			shape = predictor(gray, rects[0])
			shape = face_utils.shape_to_np(shape)
			# nose, chin,left eye lc, right eye rc left mouth corner, right mouth corner
			ans = []
			# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				# clone the original image so we can draw on it, then
				# display the name of the face part on the image
				clone = image.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)
				# loop over the subset of facial landmarks, drawing the
				# specific face part
				
				for (x, y) in shape[i:j]:
					cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				roi = image[y:y + h, x:x + w]
				if name == "nose":
					ans.append([x+w/2,y+h/2])
				elif name == "jaw":
					ans.append([x+w/2, y+h])
				elif name == "mouth":
					ans.append([x,y])
					ans.append([x+w,y+h])
				elif name == "right_eye":
					ans.append([x+w,y+h/2])
				elif name == "left_eye":
					ans.append([x,y+h/2])
				
				# show the particular face part
				
				# cv2.imshow("Image", clone)
				# cv2.waitKey(1)
			image_pt = []
			####### Append image points #########
			image_pt.append(ans[4])
			image_pt.append(ans[5])
			image_pt.append(ans[3])
			image_pt.append(ans[2])
			image_pt.append(ans[0])
			image_pt.append(ans[1])
			########## Points in order ##########
			image_points = getArray(image_pt)
			#print(image_points,size)
			yaw,roll,pitch = getYRP(image_points,size)
			temp = []
			temp.extend([yaw, roll, pitch])
			ypr.append(temp)
			feature_vec.append(image_pt)
		else:
			return []
	single_vec = prepare_vector(ypr, feature_vec)
	return single_vec
#give path name for video directory

pathIn = "/Users/anumehaagrawal/Documents/Course_Work/MP/face_detect/face_landmarks/Videos"
paths = os.listdir(pathIn)
#paths = [ "P1.avi"]
pathD = "/Users/anumehaagrawal/Documents/Course_Work/MP/face_detect/face_landmarks/Results"
done = os.listdir(pathD)
print(paths)
#done = ["P8.avi", "P58.avi", "P59.avi", "P60.avi","P64.avi","P62.avi","P63.avi","P77.avi","P76.avi", "P65.avi", "P66.avi", "P67.avi", "P71.avi", "P70.avi", "P72.avi", "P73.avi", "P74.avi"]

for i in paths:
	vect = []
	smiles = []
	if 'out_face'+i+'.csv' in done:
		continue
	print('Processing for ' + str(i))
	vec = generateVideoFeatures(pathIn+"/"+i)
	vect.append([i] + vec)
	my_df = pd.DataFrame(vect)
	my_df.to_csv('Results/out_face'+i+'.csv', index=False, header=False)
	df = pd.DataFrame(smiles)
	df.to_csv('Results/smiles'+i+'.csv', index=False, header=False)
	print('Done' + str(i))
