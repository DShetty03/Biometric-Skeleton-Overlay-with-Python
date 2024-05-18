import cv2

# initialize webcam (device 0)
def init_webcam():
    return cv2.VideoCapture(0)

#Display frame in a window
def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)

#check if the q key has been pressed to allow the user to quit the app gracefully  
def check_quit():
    return cv2.waitKey(1) & 0xFF == ord('q') #checks if 'q' is pressed. Compares result of waiting for a key press for 1 millisecond and ensures result is a standardized ASCII value to 'q'

#Release webcame and destroy all OpenCV windows
def release_webcam(cap):
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__': #check if the script is being run. common python idiom to allow/prevent parts of code from being run when modules are imported
    cap = init_webcam()
    while cap.isOpened(): # capture frame by frame from webcame
        ret, frame = cap.read()
        if not ret: 
            break
        
        #display the frame
        show_frame('Webcam Feed', frame)
        
        #check if 'q' is pressed to quit
        if check_quit():
            break
        
    #Release webcam and close display window
    release_webcam(cap)
        