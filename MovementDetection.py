# Import necessary libraries
# Make sure you install cv2 and imutils using pip before running this program
import cv2
import imutils
import time

# Initialize camera object
cam = cv2.VideoCapture(0) 

# Allow camera to warm up
time.sleep(1) 

# Initialize variables
firstFrame = None
area = 500 # Minimum area to detect movement

# Infinite loop for video processing
while True: 
    # Capture frame-by-frame
    ret, img1 = cam.read() 
    
    # Default status
    text = "Normal"
    
    # Check if frame is captured successfully
    if not ret: 
        continue
    
    # Resize frame for faster processing
    img1 = imutils.resize(img1, width=200) 
    
    # Convert frame to grayscale
    grayImg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    
    # Apply Gaussian blur to reduce noise
    gaussianBlurImg = cv2.GaussianBlur(grayImg, (21,21), 0) 
    
    # Capture the first frame
    if firstFrame is None: 
        firstFrame = gaussianBlurImg
        continue
    
    # Calculate absolute difference between the current frame and the first frame
    imgDiff = cv2.absdiff(firstFrame, gaussianBlurImg) 
    
    # Apply threshold to identify moving objects
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] 
    
    # Dilate the thresholded image to fill gaps
    threshImg = cv2.dilate(threshImg, None, iterations=2) 
    
    # Find contours in the thresholded image
    contours = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    contours = imutils.grab_contours(contours) 
    
    # Loop over the contours
    for c in contours: 
        # If contour area is smaller than the specified area, skip
        if cv2.contourArea(c) < area: 
            continue
        
        # Compute bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c) 
        
        # Draw rectangle around the moving object
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        
        # Update status
        text = "Moving Object detected!"
        print(text) 
    
    # Put status text on the frame
    cv2.putText(img1, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
    
    # Display the frame
    cv2.imshow("Camera Feed", img1) 
    
    # Check for key press
    key = cv2.waitKey(10) 
    if key == 27: # 'ESC' key to exit
        break

# Release the camera and close all windows
cam.release() 
cv2.destroyAllWindows()
