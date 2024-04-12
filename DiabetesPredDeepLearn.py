# Import necessary libraries
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the input data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the model architecture
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=10, validation_data=(x_test, y_test))

# Evaluate the model on test data
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Save the trained model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

# Testing Part
import numpy as np

# Load the model architecture from JSON
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create the model from the loaded architecture
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights("model.h5")
print("Loaded model from disk")

# Predict classes using the model
predictions = model.predict(x_test)
rounded_predictions = np.round(predictions)  # Round probabilities to get classes

# Print predictions and expected values
for i in range(len(x_test)):
    print('%s => %d (expected %d)' % (x_test[i].tolist(), rounded_predictions[i], y_test[i]))
