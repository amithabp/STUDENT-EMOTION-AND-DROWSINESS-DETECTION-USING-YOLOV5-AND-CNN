import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained CNN model
model = load_model('model2.h5')

def ff():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (48, 48))
        grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        input_data = np.reshape(grayscale_frame, (1, 48, 48, 1)) / 255.0

        # Predict emotion using CNN model
        prediction = model.predict(input_data)[0]
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotion = emotion_labels[np.argmax(prediction)]

        # Display the emotion label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, emotion, (10, 50), font, 1, (0, 255, 0), 2)

        cv2.imshow("EMOTION DETECTOR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
