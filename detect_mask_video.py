from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from send_policy import SendPolicy
from tracker import EuclideanDistTracker
import numpy as np
import imutils
import cv2
import os

class Inspector():
	def __init__(self):
		"""
		Initializes Inpector variables and objects.

		Returns:
			frame: frame of streaming video.
			faceNet: default facenet used to detect faces.
			maskNet: model trained used to detect.
		"""
		# instantiate policy to send packet
		print("[INFO] Starting send policy...")
		self.sp = SendPolicy()

		print("[INFO] Starting euclidean distance tracker...")
		self.current_face = -1
		self.tracker = EuclideanDistTracker()

		print("[INFO] Starting face/maskNet...")
		# load our serialized face detector model from disk
		prototxtPath = f"face_detector{os.sep}deploy.prototxt"
		weightsPath = f"face_detector{os.sep}res10_300x300_ssd_iter_140000.caffemodel"
		self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
		# load the face mask detector model from disk
		self.maskNet = load_model(f"face_detector{os.sep}mask_detector.model")

		# initialize the video stream
		print("[INFO] Starting video stream...")
		self.vs = VideoStream(src=0).start()


	def detect_and_predict_mask(self):
		"""
		Function to detect and return prediction of a face using or not mask.
		"""
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = self.frame.shape[:2]
		blob = cv2.dnn.blobFromImage(self.frame, 1.0, (224, 224),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		self.faceNet.setInput(blob)
		detections = self.faceNet.forward()

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		self.locs = []
		self.preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence of 50%
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = self.frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				self.locs.append((startX, startY, endX, endY))


		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype="float32")
			self.preds = self.maskNet.predict(faces, batch_size=32)


	def live(self):
		"""
		This brings Inspector to life, looping over the frames from the video
		stream
		"""
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			self.frame = self.vs.read()
			self.frame = imutils.resize(self.frame, width=400)

			# detect faces in the frame and determine if they are wearing a
			# face mask or not
			self.detect_and_predict_mask()

			# loop over the detected face locations and their corresponding
			# locations
			boxes_ids = self.tracker.update(self.locs)
			#print(boxes_ids)
			for (box, pred) in zip(boxes_ids, self.preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY, id_registered) = box
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw
				# the bounding box and text, while also encoding bytes from
				# the most recent frame
				if mask > withoutMask:
					label = "Mask"
					image_bytes = None
				else:
					label = "No Mask"
					face_detected = self.frame[startY:endY, startX:endX]
					image_bytes = cv2.imencode('.jpg', face_detected)[1].tobytes()

				print(self.current_face, id_registered)
				if self.current_face < id_registered:
					self.current_face = id_registered
					# interacts with API
					self.sp.send(image_bytes)


				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(self.frame, str(id_registered), (startX, startY - 25),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
				cv2.putText(self.frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(self.frame, (startX, startY), (endX, endY), color, 2)

			# show the output frame
			cv2.imshow("Inspector ConEg", self.frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# do a bit of cleanup
		cv2.destroyAllWindows()
		self.vs.stop()


if __name__ == "__main__":
	insp = Inspector()
	insp.live()