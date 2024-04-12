# Import necessary libraries
import cv2
import os

# Define directory and file name
ds = "Images"
name = "FaceDetection"
# I've created a Folder "Images" in the same directory as the program. "Images" is having a subfolder called "FaceDetection"

# Check if parent directory exists, if not, create it
if not os.path.isdir(ds): 
    os.mkdir(ds) 

# Define path for the subdirectory
path = os.path.join(ds, name) 

# Create subdirectory if it does not exist
if not os.path.isdir(path): 
    os.mkdir(path) 

# Define width and height for resizing
(width, height) = (500, 550) 

# Load Haar cascade classifier for face detection
alg = 'haarcascade_frontalface_default.xml'
# We can download the 'haarcascade_frontalface_default.xml' from internet and paste the file to the program's directory. Otherwise, we can give the path of the downloaded xml file.   
haar_cascade = cv2.CascadeClassifier(alg) 

# Initialize camera object
cam = cv2.VideoCapture(0) 

# Initialize count for image naming
count = 1

# Capture images
while count <= 30: 
    # Print current count
    print(count) 
    
    # Capture frame-by-frame
    ret, img = cam.read() 
    
    # Check if frame is captured successfully
    if not ret: 
        continue
    
    # Convert frame to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Detect faces in the grayscale frame
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) 
    
    # Loop over detected faces
    for (x, y, w, h) in face: 
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2) 
        
        # Extract face region
        faceOnly = grayImg[y:y+h, x:x+w] 
        
        # Resize face region
        resizedImg = cv2.resize(faceOnly, (width, height)) 
        
        # Save resized face region
        cv2.imwrite("%s/%s.jpg" % (path, "Face"+str(count)), resizedImg) 
        
        # Increment count
        count += 1
    
    # Display frame with detected faces
    cv2.imshow("Face Detection", img) 
    
    # Check for key press
    key = cv2.waitKey(10) 
    if key == 27: # 'ESC' key to exit
        break

# Print success message
print("Image Captured Successfully") 

# Release the camera and close all windows
cam.release() 
cv2.destroyAllWindows()
