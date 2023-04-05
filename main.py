import cv2
import pathlib

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0) # Use built in camera

while True:
    _, frame = camera.read() # Read from camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,width,height) in faces:
        # draw onto frame a rectangle starting at x,y to x+width, y+height
        # of color 255,255,0 in BGR, width line width 2
        cv2.rectangle(frame,(x,y), (x+width,y+height), (255,255,0),2)

    cv2.imshow("Faces",frame) 

    # Quit the program once we press button 'q'
    if cv2.waitKey(1) == ord('q'): 
        break 

# Close window and camera after quitting program
camera.release()
cv2.destoryAllWindows()
