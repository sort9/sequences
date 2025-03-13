import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from src import lists

# For this example, I will use list1 as the sequence for training
data = np.array(lists.list1, dtype=int).reshape((len(lists.list1), 1))  # Input sequence
target = np.power(data, 2)  # Target sequence (squares of input numbers)

# Normalize or reshape if necessary for LSTM
data = data.reshape((data.shape[0], 1, 1))  # Reshaping for LSTM (samples, time steps, features)

# Build the model
model = Sequential()
model.add(LSTM(32, input_shape=(data.shape[1], data.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(data, target, epochs=200, batch_size=1, verbose=1)

# Predict the next number in the sequence based on incompletelist
incomplete_data = np.array(lists.incompletelist[-1], dtype=int).reshape((1, 1, 1))  # Last value in incompletelist
predicted_value = model.predict(incomplete_data)
print(f"Predicted next number: {predicted_value[0][0]}")