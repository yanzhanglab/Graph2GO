from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
import numpy as np


def train_nn(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    model.add(Dense(Y_train.shape[1],activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0)

    y_prob = model.predict(X_test)
    return y_prob





    


