import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model_path = "plant_classifier_transfer_model.h5"  # Replace with the path to your saved model
model = load_model(model_path)

# Define class names (mapping from class_0, class_1, etc. to actual labels)
class_labels = ['anthurium', 'clivia', 'dieffenbachia', 'dracaena', 'gloxinia', 
                'kalanchoe', 'orchid', 'sansevieria', 'violet', 'zamioculcas']

# Parameters
img_height, img_width = 224, 224  # Image dimensions used during training

# Initialize the video capture (0 for the default laptop camera)
cap = cv2.VideoCapture(0)

def predict_frame(frame):
    """
    Preprocesses the frame and predicts the class using the trained model.
    Args:
        frame: A single frame captured from the video feed.
    Returns:
        str: Predicted class label.
        float: Confidence score of the prediction.
    """
    # Resize the frame to the input size of the model
    resized_frame = cv2.resize(frame, (img_height, img_width))
    
    # Preprocess the frame
    preprocessed_frame = preprocess_input(resized_frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index] * 100  # Convert to percentage
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label, confidence_score

print("Press 'q' to quit the application.")

# Real-Time Object Detection
while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break

    # Flip the frame horizontally for natural mirroring
    frame = cv2.flip(frame, 1)
    
    # Predict the class
    predicted_label, confidence = predict_frame(frame)

    # Display the predicted label and confidence on the frame
    text = f"Prediction: {predicted_label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame in a window
    cv2.imshow("Real-Time Plant Detection", frame)

    # Print the predicted label and confidence to the console
    print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}%")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
