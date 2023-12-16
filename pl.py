import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers, models

def preprocessing(features, labels):
    #expects grayscale images
    num_samples, image_rows, image_columns = features.shape
    # convert to float
    features = features.astype('float32')
    
    # reshape to 1 channel
    features = features.reshape(num_samples, image_rows, image_columns, 1)
    
    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Split the temporary set into validation and test sets
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    #scaling
    scaler = StandardScaler()
    X_train = X_train.reshape((X_train.shape[0], image_rows * image_columns))
    X_validation = X_validation.reshape((X_validation.shape[0], image_rows * image_columns))
    X_test = X_test.reshape((X_test.shape[0], image_rows * image_columns))
    
    X_train = scaler.fit_transform(X_train)
    X_validation= scaler.transform(X_validation)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.reshape((X_train.shape[0], image_rows, image_columns, 1))
    X_validation = X_validation.reshape((X_validation.shape[0], image_rows, image_columns, 1))
    X_test = X_test.reshape((X_test.shape[0], image_rows, image_columns, 1))

    return X_train, X_test, X_validation, y_train, y_test, y_validation

def baseline_model(X_train, X_test, X_validation, y_train, y_test, y_validation):
    dummy_classifier = DummyClassifier(strategy="most_frequent") #using "most frequent" strategy
    dummy_classifier.fit(X_train, y_train)

    y_val_pred = dummy_classifier.predict(X_validation)
    val_accuracy = accuracy_score(y_validation, y_val_pred)
    print(f"Accuracy on validation set: {val_accuracy:.4f}")
    
    y_pred = dummy_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")
    
def linear_model(X_train, X_test, X_validation, y_train, y_test, y_validation):
    num_samples, num_channels, height, width = X_train.shape
    X_train_flattened = X_train.reshape((num_samples, num_channels * height * width))
    num_samples_test, _, _, _ = X_test.shape
    X_test_flattened = X_test.reshape((num_samples_test, num_channels * height * width))
    X_validation_flattened = X_validation.reshape((X_validation.shape[0], num_channels * height * width))   
    
    logistic_regression = LogisticRegression(max_iter=1000, C=1)
    logistic_regression.fit(X_train_flattened, y_train)
    
    # Evaluate on the Validation Set
    y_val_pred = logistic_regression.predict(X_validation_flattened)
    val_accuracy = accuracy_score(y_validation, y_val_pred)
    print(f"Accuracy on validation set: {val_accuracy:.4f}")

    # Evaluate on the Test Set
    y_test_pred = logistic_regression.predict(X_test_flattened)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy on test set: {test_accuracy:.4f}")
    
def cnn(X_train, X_test, X_validation, y_train, y_test, y_validation, num_classes):
    #print(X_train[0])
    # Create a simple CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with 10 classes

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_validation, y_validation))
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    #evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
def pipeline(features, labels, num_classes):
    X_train, X_test, X_validation, y_train, y_test, y_validation = preprocessing(features, labels)
    print("-------------------")
    print("Baseline model:")
    baseline_model(X_train, X_test, X_validation, y_train, y_test, y_validation)
    print("-------------------")
    print("Linear model:")
    linear_model(X_train, X_test, X_validation, y_train, y_test, y_validation)
    print("-------------------")
    print("CNN model:")
    cnn(X_train, X_test, X_validation, y_train, y_test, y_validation, num_classes=num_classes)
    print("-------------------")