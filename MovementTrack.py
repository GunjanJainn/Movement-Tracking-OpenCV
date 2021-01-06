# Movement tracking

import cv2

cap= cv2.VideoCapture("Walking.mp4")
#reading each frame of the video
ret, frame1= cap.read()
ret, frame2= cap.read()

while cap.isOpened():
    #Finding a difference in consecutive frames of the video to detect any changes
    diff= cv2.absdiff(frame1, frame2)
    #Grayscaling the difference found
    gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #Refining the image for better results
    blur= cv2.GaussianBlur(gray, (5,5), 0)
    #if needed, use a trackbar to know the exact value of threshold required
    _, thresh= cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated= cv2.dilate(thresh, None, iterations= 3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 255), 3)
    for con in contours:
        (x, y, w, h) = cv2.boundingRect(con)
        if cv2.contourArea(con) < 900:  
            #you may use any other number than 900, depending on the video you use
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2 )
        cv2.putText(frame1, "Status: {}" .format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    
    cv2.imshow("Feed", frame1)
    #for maintaining a continuous loop of frames
    frame1= frame2
    ret, frame2= cap.read()
    #to exit the loop, press the space bar (ascii code= 32)
    if cv2.waitKey()== 32:
        break
cv2.destroyAllWindows()
cap.release()