import cv2
import numpy as np
import matplotlib.pylab as plt

#define the region to start counting
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#count of vehicle on left side of road
def left(dilated1):

    region_of_interest_vertices = [
        (33, 688),
        (33, 557),
        (310, 461),
        (500, 461),
    ]
    cropped_image = region_of_interest(dilated1, np.array([region_of_interest_vertices], np.int32),)

    no_of_vehicle1 = 0

    #to find the contours
    contours, _ = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        #draw rectangle if area is greater than 2500
        if h*w > 2500:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            no_of_vehicle1 = no_of_vehicle1+1

    return no_of_vehicle1

#count of vehicle on right side of road
def right(dilated1):

    region_of_interest_vertices = [
        (370, 690),
        (608, 461),
        (803, 461),
        (899, 690),
    ]
    cropped_image = region_of_interest(dilated1, np.array([region_of_interest_vertices], np.int32),)

    no_of_vehicle2 = 0

    contours, _ = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h*w > 2500:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            no_of_vehicle2 = no_of_vehicle2+1
            
    return no_of_vehicle2

fourcc = cv2.VideoWriter_fourcc(* 'XVID')
out = cv2.VideoWriter('output.mp4',fourcc,24.0,(720,1280))  
cap = cv2.VideoCapture('input_video.mp4')

#background subtaction
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

ret, frame1 = cap.read()
ret, frame2 = cap.read()

a = 0
count1 = 0
b = 0
count2 = 0

while cap.isOpened():
	#applying bg subtraction on 2 frames 
    fgmask = fgbg.apply(frame1)
    fgmask2 = fgbg.apply(frame2)

    #finding difference in the frames
    diff = cv2.absdiff(fgmask, fgmask2)

    kernel = np.ones((15,15), np.uint8)
    #applying threshold
    _, thresh = cv2.threshold(diff, 107, 255, cv2.THRESH_BINARY)
    #dilating 
    dilated = cv2.dilate(thresh, kernel, iterations=1)   

    left_no = left(dilated)
    right_no = right(dilated)

    cv2.putText(frame1, "Left: " , (20, 55), 3, 1, (170,255,255), 2)
    cv2.putText(frame1, "Vehicles Detected: " + str(left_no), (20, 85), 3, 0.9, (170,255,255), 2)
    if a != left_no and left_no != 0 and a<left_no:
            count1 = count1 + 1
    a = left_no
    cv2.putText(frame1, "Vehicles Passed: " + str(count1), (20, 115), 3, 0.9, (170,255,255), 2)

    cv2.putText(frame1, "Right: ", (950, 55), 3, 1, (170,255,255), 2)
    cv2.putText(frame1, "Vehicles Detected: " + str(right_no), (950, 85), 3, 0.9, (170,255,255), 2)
    if b != right_no and right_no != 0 and b<right_no:
            count2 = count2 + 1
    b = right_no
    cv2.putText(frame1, "Vehicles Passed: " + str(count2), (950, 115), 3, 0.9, (170,255,255), 2)

    cv2.putText(frame1, "TRAFFIC COUNTER" , (500, 31), 4, 1, (0, 0, 0), 2)
    cv2.putText(frame1, "- SWAGATIKA" , (550, 61), 4, 0.7, (0, 0, 0), 2)
    img = cv2.line(frame1, (608, 461), (803, 461), (255,0,0), 2)
    img = cv2.line(frame1, (310, 461), (510, 461), (255,0,0), 2)

    out.write(frame1)
    cv2.imshow("vid", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    #Press 'q' key to exit 
    if cv2.waitKey(50) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()