import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('model/fer2013.csv')

# Count the number of occurrences of each emotion in the dataset
emotion_counts = df['emotion'].value_counts()

# Plot the distribution of the emotions
plt.figure(figsize=(8, 6))
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xticks(emotion_counts.index, ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
plt.xlabel('Emotion')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Emotions in FER2013 Dataset')
plt.show()
