import cv2

class VideoRecorder:
    def __init__(self, filename='output.avi', fps=20.0, frame_size=(640, 480)):
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.recording = False  # Updated attribute name
        self.out = None

    def start_recording(self):
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)
            self.recording = True

    def stop_recording(self):
        if self.recording:
            self.out.release()
            self.recording = False

    def record_frame(self, frame):
        if self.recording:
            self.out.write(frame)

    def is_recording(self):
        return self.recording
