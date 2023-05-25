import keras
import json

# Load the model from the .h5 file
model = keras.models.load_model('model2.h5')

# Convert the model to a JSON string
model_json = model.to_json()

# Write the JSON string to a file
with open('model2.json', 'w') as json_file:
    json_file.write(model_json)
