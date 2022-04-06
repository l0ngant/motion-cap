# Importing libraries:
import time
from datetime import datetime
import cv2
import pandas

# Starting video capture on webcam:
vid = cv2.VideoCapture(0)
first_frame = None

# Creating a list that holds the status changes timestamps:
status_list = [None, None]
timestamps = []
df = pandas.DataFrame(columns=["Motion start", "Motion end"])

# Main script loop:
while True:
    check, frame = vid.read()
    status = 0 # This is the status of the image: 0 means there's no movement, 1 means there's movement
    # The frame is reduced to grayscale and blurred to increase accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # The first iteration saves the first frame, the next ones go back
    # to the beginning of the while loop with the "continue" line
    if first_frame is None:
        first_frame = gray
        continue
    # Calculates the difference between the first frame and the current one ("gray")
    delta_frame = cv2.absdiff(first_frame, gray)
    # Applies a binary threshold to the delta image, meaning it colors white - 255
    # the areas of the image where there's more than 30 units of color intensity difference
    # between the first frame and the current frame. The method cv2.threshold
    # returns a tuple, we access its 2nd value
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # Dilates the white areas to smoothen the image
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    # Find the contours of the white areas and if they are wider than a certain
    # amount of pixels, the areas are considered moving objects
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 6000:
            continue
        status = 1 # Here the status changes, the cam detects movement!
        (x, y, h, w) = cv2.boundingRect(contour)
        # Draw the rectangle on the original color image,
        # coordinates belong to the top left and bottom right points 
        # of the motion areas:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    # Empties the list keeping only the 2 last elements:
    status_list = status_list[-2:]
    if status_list[-1] == 1 and status_list[-2] == 0:
        timestamps.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        timestamps.append(datetime.now())    

    # Showing the capture window:
    cv2.imshow('Identifying moving objects... press q to close', frame)
    key = cv2.waitKey(1)
    # Closing if 'q' is pressed:
    if key == ord('q'):
        if status == 1:
            timestamps.append(datetime.now())
        break
for i in range(0, len(timestamps), 2):
    df = df.append({"Motion start":timestamps[i], "Motion end":timestamps[i+1]}, ignore_index=True)
# Exporting the timestamps lists to a .csv file:
df.to_csv('Times.csv', index=False)
# Release the cam, close the script window:
vid.release()
cv2.destroyAllWindows