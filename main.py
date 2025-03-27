import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class SignLanguageRecognizer:
    def __init__(self, model_path):
        """
        Initialize the sign language recognizer
        
        Args:
        model_path (str): Path to the pre-trained TensorFlow CNN model
        """
        try:
            # Load the pre-trained model
            self.model = load_model(model_path)
            
            # Define class labels (modify this list based on your trained model)
            self.class_labels = [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                'Y', 'Z'
            ]
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_hand_image(self, hand_img):
        """
        Preprocess the hand image for sign language recognition
        
        Args:
        hand_img (numpy.ndarray): Input hand image
        
        Returns:
        numpy.ndarray: Preprocessed image of shape (1, 28, 28, 1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize and reshape to (1, 28, 28, 1)
        normalized = resized / 255.0
        processed = normalized.reshape(1, 28, 28, 1)
        
        return processed
    
    def predict_sign(self, processed_image):
        """
        Predict the sign language letter
        
        Args:
        processed_image (numpy.ndarray): Preprocessed hand image
        
        Returns:
        str: Predicted sign language letter
        """
        try:
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Get the index of the highest probability
            predicted_class_index = np.argmax(predictions[0])
            
            # Get the predicted class label
            predicted_class = self.class_labels[predicted_class_index]
            
            # Get the confidence score
            confidence = predictions[0][predicted_class_index]
            
            return predicted_class, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None

def main():
    # Path to your pre-trained TensorFlow model
    # IMPORTANT: Replace this with the actual path to your trained model
    MODEL_PATH = 'sign_language_model.h5'
    
    # Initialize the recognizer
    try:
        recognizer = SignLanguageRecognizer(MODEL_PATH)
    except Exception as e:
        print(f"Failed to initialize recognizer: {e}")
        return
    
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Define the region of interest (ROI) box parameters
    box_size = 200  # Size of the square box
    box_color = (0, 255, 0)  # Green color
    box_thickness = 2
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate box coordinates (centered)
        x_start = (width - box_size) // 2
        y_start = (height - box_size) // 2
        
        # Draw the box for hand placement
        cv2.rectangle(frame, 
                      (x_start, y_start), 
                      (x_start + box_size, y_start + box_size), 
                      box_color, 
                      box_thickness)
        
        # Extract the hand region
        hand_roi = frame[y_start:y_start+box_size, x_start:x_start+box_size]
        
        # Display the frame with box
        cv2.imshow('Hand Capture for Sign Language', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Process and predict when 'p' is pressed
        if key == ord('p'):
            try:
                # Preprocess the hand image
                processed_hand = recognizer.preprocess_hand_image(hand_roi)
                
                # Predict the sign
                predicted_sign, confidence = recognizer.predict_sign(processed_hand)
                
                if predicted_sign:
                    # Display prediction on the frame
                    prediction_text = f"Predicted Sign: {predicted_sign} (Confidence: {confidence:.2%})"
                    cv2.putText(frame, prediction_text, 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 255, 0), 
                                2)
                    
                    print(prediction_text)
            except Exception as e:
                print("Error processing hand image:", e)
        
        # Exit on 'q' key press
        elif key == ord('q'):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()