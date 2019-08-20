from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np


def train_nn(X_train, Y_train, X_test, Y_test, ontology):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y_train.shape[1],activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0)

    y_prob = model.predict(X_test)
    return y_prob





    
