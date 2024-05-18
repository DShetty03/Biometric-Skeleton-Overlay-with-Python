import cv2
import mediapipe as mp
import utils
import numpy as np

class PoseDetector:
    def __init__(self): # The __init__ method is the constructor for the PoseDetector class. It initializes the necessary components from the Mediapipe library to perform pose detection and draw the landmarks.
        
        # Initialize BlazePose Full for more accurate 3D pose estimation
        
        #import the pose module from Mediapipe. 
        # mp.solutions.pose provides the necessary tools for pose detection.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                    model_complexity=2,  # Use complexity level 2 for more accuracy
                                    enable_segmentation=True,
                                    min_detection_confidence=0.7,
                                    min_tracking_confidence=0.7)
        #imports the drawing utilities from Mediapipe. 
        # mp.solutions.drawing_utils provides functions to draw the detected landmarks on images. 
        # By storing it in self.mp_drawing, you can use these drawing functions to visualize the results.
        self.mp_drawing = mp.solutions.drawing_utils
        self.smoother = utils.PoseSmoother(alpha = 0.75)
    
    #process a given frame to detect human poses and annotate the frame with detected landmarks    
    def detect_pose(self,frame):
        #convert frame to RGP as mediapipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #process frame and get result with pose landmarks
        result = self.pose.process(rgb_frame)
        
        # Create a black background
        black_frame = np.zeros_like(frame)
        
        #if landmarks detected, draw on the frame
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                utils.get_drawing_spec(), utils.get_drawing_spec()
            )
            self.mp_drawing.draw_landmarks(
                black_frame,result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                utils.get_drawing_spec(), utils.get_drawing_spec()
            )
        return frame, black_frame, result
        