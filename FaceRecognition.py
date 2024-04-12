# Import necessary libraries
import cv2
import numpy as np
import os

# Load Haar cascade classifier for face detection
haar_file = 'haarcascade_frontalface_default.xml'
# Download the 'haarcascade_frontalface_default.xml' file from internet and paste it in the program's directory or paste the path of the file in between the quotes in the above line.

face_cascade = cv2.CascadeClassifier(haar_file)

# Define directory containing face image datasets
datasets = 'datasets'
# I have training data in a folder named "datsets", in that directory each sub-directory contains a training sample, i.e., one face training data in each sub-directory.
# To generate the training dataset, you can prefer 'FaceDetection.py' in my repo.
# Print status message
print('Training...')

# Initialize variables for training
(images, labels, names, id) = ([], [], {}, 0)

# Loop through subdirectories in the dataset directory
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        # Assign ID to each subject
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        # Loop through files in the subdirectory
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            # Read images and corresponding labels
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

# Convert images and labels to numpy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]
print(images, labels)                   

# Define width and height for resizing
(width, height) = (130, 100)

# Initialize LBPH Face Recognizer model
model = cv2.face.LBPHFaceRecognizer_create()
# We can also use FisherFace model

# Train the model
model.train(images, labels)

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Initialize counter
cnt = 0

# Face recognition loop
while True:
    (a, im) = webcam.read()
    if(not a):
        print("unable to read input from camera")
        break
    # Convert frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 255, 0), 2)
        # Extract face region
        face = gray[y:y + h, x:x + w]
        # Resize face region
        face_resize = cv2.resize(face, (width, height))
        # Predict the label of the face
        prediction = model.predict(face_resize)
        # Draw rectangle around detected face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Check if prediction confidence is below a threshold
        if prediction[1] < 800:
            # Display recognized name and confidence level
            cv2.putText(im, '%s - %.0f' % (names[int(prediction[0])], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[int(prediction[0])])
            cnt = 0
        else:
            # Increment counter for unknown person
            cnt += 1
            # Display "Unknown" if confidence level is above threshold
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # If unknown for consecutive frames, save the frame as "unKnown.jpg"
            if(cnt > 100):
                print("Unknown Person")
                cv2.imwrite("unKnown.jpg", im)
                cnt = 0
    # Display frame with face recognition
    cv2.imshow('FaceRecognition', im)
    # Check for key press to exit
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
