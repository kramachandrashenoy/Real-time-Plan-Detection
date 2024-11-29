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

def predict_image(image):
    """
    Preprocesses the captured image and predicts the class using the trained model.
    Args:
        image: Captured image.
    Returns:
        str: Predicted class label.
        float: Confidence score of the prediction.
    """
    # Resize the image to the input size of the model
    resized_image = cv2.resize(image, (img_height, img_width))
    
    # Preprocess the image
    preprocessed_image = preprocess_input(resized_image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index] * 100  # Convert to percentage
    predicted_class_label = class_labels[predicted_class_index]
    
    # Debug: Print all probabilities
    print("Class Probabilities:", predictions[0])

    return predicted_class_label, confidence_score

# Initialize the video capture (0 for the default laptop camera)
cap = cv2.VideoCapture(0)

print("Press 'Enter' to capture an image for prediction.")
print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break

    # Flip the frame horizontally for natural mirroring
    frame = cv2.flip(frame, 1)
    
    # Display the frame
    cv2.imshow("Real-Time Plant Detection", frame)

    # Check for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the application when 'q' is pressed
        break
    elif key == 13:  # Enter key to capture the image
        # Predict the class of the captured frame
        predicted_label, confidence = predict_image(frame)

        # Display the prediction result
        print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}%")
        
        # Overlay prediction result on the frame
        cv2.putText(frame, f"Prediction: {predicted_label} ({confidence:.2f}%)", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the captured image with the prediction
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)  # Wait for any key before continuing

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
