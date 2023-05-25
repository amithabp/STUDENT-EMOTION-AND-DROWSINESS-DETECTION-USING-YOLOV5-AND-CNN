import pandas as pd
import numpy as np


data = pd.read_csv('model/fer2013.csv')


data = data.sample(frac=1, random_state=42)


train_data = data.iloc[:25000, :]
val_data = data.iloc[25000:30000, :]
test_data = data.iloc[30000:, :]


train_data.to_csv('trainfer2013.csv', index=False)
val_data.to_csv('valfer2013.csv', index=False)
test_data.to_csv('testfer2013.csv', index=False)