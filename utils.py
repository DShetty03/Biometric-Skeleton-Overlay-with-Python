import mediapipe as mp

def get_drawing_spec():
    # Customize the drawing specification for landmarks
    return mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

#smoothing function
#Smoothing: Technique used to reduce noise and fluctuations in data. Making it easier to observe trends and patterns
# Smoothing helps stabilize detected landmarks over time, resulting in more consistent and less jittery pose estimation.
# Why Smoothing is Needed?: noise reduction and stabilization 
#Types of Smoothing:
    # Moving Average: 
        #calculates the average of a fixed number of recent data points. It's simple but effective for reducing short-term fluctuations.
        # Example: For a window size of 3, the smoothed value at time t would be the average of values at times t-2, t-1, and t.
    # Exponential Smoothing: 
        # gives exponentially decreasing weights to older observations. It is more responsive to recent changes compared to the moving average.
        #Example: The new smoothed value is a weighted average of the previous smoothed value and the current observation.
        # formula: For a series of observations x1, x2, x3, ... xt, the exponentially smoothed series s_t is:
            # s_t = (α)x_t + (1-α)s_(t-1)
                # α = smoothing factor (0<α<1)
                #x_t = current observation
                #s_(t-1) is prev smoothed value
        # in pose detection, x_t = current frame's x-coordinate, s_(t-1) = prev frame smooth coordinate, α = smoothing factor
    # Kalman Filter:
        # A more advanced technique that not only smooths data but also predicts future values based on a mathematical model. It is widely used in applications requiring real-time filtering and prediction.
        # Examples of its applications:
            # Financial Modeling:
                # Stock Price Prediction: Kalman filters can be used to model and predict the movement of stock prices by filtering out noise from the observed price data.
                # Portfolio Optimization: Portfolio Optimization: Estimating the volatility and correlations of asset returns to optimize the allocation of assets in a portfolio.
            # Signal Processing: 
                # Speech Enhancement: Reduce noise in audio signals
                # Image Stabilization: Estimates true motion of camera and compensates for unwanted movements
            # Autonomous Vehicles: 
                #Sensor Fusion: Combining data from various sensors (e.g., LIDAR, cameras, radar) to accurately estimate the position and velocity of the vehicle and surrounding objects.
                # Path Planning: Estimating the future states of the vehicle to plan safe and efficient trajectories.
        

class PoseSmoother:
    def __init__(self, alpha=0.75):
        self.alpha = alpha #smoothing factor (coefficient). 0.9 = 90% of previous landmark and 10% of current are used to calculate new smoothed value
        self.previous_landmarks = None #stores landmark from previous frame. initialized to none

    #applies exponential smoothing to detected pose landmarks
    def smooth_landmarks(self, landmarks):
        if self.previous_landmarks is None: #if true, then first frame is being processed
            self.previous_landmarks = landmarks
        else: # else, processed at least one frame beofre and can continue smoothing
            for i, landmark in enumerate(landmarks.landmark): # loop through each landmark. loop iterates over each detected landmark in the current frame. 
                #exponential smoothing
                # For each coordinate (x, y, z) of the landmark, the new smoothed value is calculated using exponential smoothing:
                
                # self.alpha * self.previous_landmarks.landmark[i].x: This term gives a weight to the previous frame's landmark coordinate.
                # (1 - self.alpha) * landmark.x: This term gives a weight to the current frame's landmark coordinate.
                # Sum of both terms: This results in a new coordinate value that is a weighted average of the previous and current values, effectively smoothing the transition.
                self.previous_landmarks.landmark[i].x = self.alpha * self.previous_landmarks.landmark[i].x + (1 - self.alpha) * landmark.x
                self.previous_landmarks.landmark[i].y = self.alpha * self.previous_landmarks.landmark[i].y + (1 - self.alpha) * landmark.y
                self.previous_landmarks.landmark[i].z = self.alpha * self.previous_landmarks.landmark[i].z + (1 - self.alpha) * landmark.z
        return self.previous_landmarks