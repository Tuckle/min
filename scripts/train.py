from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pandas as pd

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_data():
    """Retrieve the dataset and process the data."""
    # Set defaults.
    nb_classes = 1
    batch_size = 64
    input_shape = (9,)

    # Get the data.     
    data = pd.read_csv('newdata.csv')
    Y = data.pop("EVENT")
    X = data.values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
    """
    nb_classes, batch_size, input_shape, x_train, \
    x_test, y_train, y_test = get_data()

    model = compile_model(network, nb_classes, input_shape)

    #print(x_train.shape)
    #print(y_train.shape)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

#nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_data()