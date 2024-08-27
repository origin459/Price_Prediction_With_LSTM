import joblib
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import TimeseriesGenerator
from dataset import fetch_and_process_data
from model import build_model

def main():
    user_key = 'your_user_key'
    secret_key = 'your_secret_key'
    
    # Fetch and process the data
    features, target = fetch_and_process_data(user_key, secret_key)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle=False)

    # Define sequence length and batch size
    win_len = 288  # Number of past time steps to consider
    batch_size = 32
    num_features = x_train.shape[1]  # Number of features

    # Create time series generators
    train_generator = TimeseriesGenerator(x_train, y_train, length=win_len, sampling_rate=1, batch_size=batch_size)
    test_generator = TimeseriesGenerator(x_test, y_test, length=win_len, sampling_rate=1, batch_size=batch_size)

    # Build the model
    model = build_model(input_shape=(win_len, num_features))

    # Define a ModelCheckpoint callback to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(train_generator, epochs=500, validation_data=test_generator, shuffle=False, callbacks=[checkpoint], verbose=1)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')

    # Save the model
    model.save('final_model.h5')

if __name__ == "__main__":
    main()
