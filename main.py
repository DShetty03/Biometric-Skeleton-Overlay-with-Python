import cv2
import numpy as np
import cam_capture
import pose_detection
import video_record

def main():
    # Initialize the webcam
    cap = cam_capture.init_webcam()
    
    # Initialize the PoseDetector object
    pose_detector = pose_detection.PoseDetector()
    
    # Initialize the VideoRecorder object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_recorder = video_record.VideoRecorder(frame_size=(frame_width, frame_height))

    while cap.isOpened():
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to detect and annotate poses
        display_frame, black_frame, result = pose_detector.detect_pose(frame)
        
        # Display the normal frame with annotations
        cam_capture.show_frame('Webcam Feed', display_frame)

        # Check if 'r' is pressed to start/stop recording
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if video_recorder.is_recording():
                video_recorder.stop_recording()
            else:
                video_recorder.start_recording()

        # If recording, write the black frame to the video file
        if video_recorder.is_recording():
            video_recorder.record_frame(black_frame)

        # Check if 'q' is pressed to quit the application
        if key == ord('q'):
            break

    # Release the webcam and close the display window
    cam_capture.release_webcam(cap)
    # Release the video writer if still open
    video_recorder.stop_recording()

if __name__ == '__main__':
    main()
