from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, LSTM, GlobalMaxPooling1D
from keras.optimizers import Adam

def build_model(input_shape):
    model = Sequential()

    # Adding LSTM layers with LeakyReLU activations and Dropout for regularization
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(GlobalMaxPooling1D())  # Reducing the dimension to a 1D output
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model