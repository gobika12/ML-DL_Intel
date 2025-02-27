import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
# Load the IMDB dataset
num_words = 10000
# Consider only the top 10,000 words in the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
max_len = 200 # Set the maximum length of sequences
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
embedding_dim = 128
hidden_units = 64
model = Sequential()
model.add(Embedding(input_dim=num_words,output_dim=embedding_dim,input_length=max_len))
model.add(Bidirectional(LSTM(hidden_units)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
