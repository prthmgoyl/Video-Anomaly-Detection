import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest

# Function to extract features from video frames
def extract_features(frame):
# Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Resize the frame to a fixed size
    resized = cv2.resize(gray, (64, 128))
# Flatten the image into a 1D vector
    flattened = resized.flatten()
    return flattened

# Function to load video frames from a directory
def load_frames(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.tif'):
            frame_path = os.path.join(directory, filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
    return frames

# Function to detect anomalies in a video using Isolation Forest
def detect_anomalies(frames):
# Extract features from video frames
    features = [extract_features(frame) for frame in frames]

# Convert features to numpy array
    X = np.array(features)
    
# Create and fit the Isolation Forest model
    model = IsolationForest(contamination=0.1)
    model.fit(X)
    
# Predict anomaly scores for the frames
    anomaly_scores = model.decision_function(X)
    
    return anomaly_scores

# Function to draw a red rectangle around anomaly pixels
def draw_rectangle(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


# Load frames from the test video sequence
frames = load_frames('UCSDped1/Test/Test001')

# Detect anomalies in the video sequence
anomaly_scores = detect_anomalies(frames)

# Display the video frames along with anomaly scores and red rectangles
for frame, score in zip(frames, anomaly_scores):
# Rescale the anomaly score to the range [0, 255]
    score_rescaled = int(255 * (score - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores)))
    
# Convert the score to a color (blue = low anomaly, red = high anomaly)
    color = (255 - score_rescaled, 0, score_rescaled)
    
# Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# Threshold the grayscale frame to create a binary mask of anomaly pixels
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
# Draw the red rectangle around anomaly pixels
    draw_rectangle(frame, mask)
    
# Draw the anomaly score as text on the frame
    cv2.putText(frame, f"Anomaly Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
# Display the frame
    cv2.imshow("Video Frame", frame)
    
# Wait for a key press and check if the user wantsto exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the OpenCV window
cv2.destroyAllWindows()

