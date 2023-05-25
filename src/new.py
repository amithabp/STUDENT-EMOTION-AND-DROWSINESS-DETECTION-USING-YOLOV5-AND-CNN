import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the pre-trained model
model = load_model('src/model/my_model.h5')

# Define the emotion classes
emotions = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}

# Start the camera
cap = cv2.VideoCapture(0)

# Set the width and height of the camera
cap.set(3, 640)
cap.set(4, 480)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame to 48x48
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Convert the resized frame to an array
        img_array = img_to_array(resized)

        # Expand the dimensions of the array
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the input image
        img_array = preprocess_input(img_array)

        # Predict the emotion of the input image
        prediction = model.predict(img_array)

        # Get the index of the emotion class with the highest probability
        emotion_index = np.argmax(prediction)

        # Get the emotion class label
        emotion_label = emotions[emotion_index]

        # Draw the emotion label on the frame
        cv2.putText(frame, emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
