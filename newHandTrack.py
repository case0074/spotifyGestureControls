import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import handTrackingModule as htm

def main():
    #creating the camera
    cap = cv.VideoCapture(0)
    
    #creating handDector from the custom module
    detector = htm.handDetector()
    
    pTime = 0  # Initialize pTime inside main()
    
    while True:
        success, img = cap.read()
        if not success:
            break  # Exit loop if no frame is captured

        img = detector.findHands(img, draw=True) #call find hands to track hands from camera
        lmList = detector.findPosition(img) #Maps the points on the fingers

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

#file