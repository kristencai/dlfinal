import tensorflow as tf
import numpy as np
from re import X
from keras.applications import ResNet50

from keras.applications.resnet import ResNet152
from keras.layers import Dense, Flatten, Dropout
from keras import Sequential
from keras.models import Model

from preprocess import get_labels, unpickle, preprocess_images

from re import X

def train_model(images, one_hots):

    # load in the pre-trained resnet
    # resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(256,256,3))
    resnet50 = ResNet152(include_top=False, weights='imagenet', input_shape=(256,256,3))
    

    # freeze the pre-trained layers, and only train the newly added layers
    # for layer in resnet152.layers:
    #   layer.trainable = False
    for layer in resnet50.layers:
        layer.trainable = False


    # add new layers
    # x = Flatten()(resnet152.output)
    # x = Dense(128, activation='relu')(x)
    # predictions = Dense(2, activation='sigmoid')(x)

    # train_model = Sequential([
    # flatten=Flatten()(resnet50.output)
    # dense1=Dense(128, activation='relu')(flatten)
    # dropout1=Dropout(rate=0.3)(dense1)
    # dense2=Dense(64, activation='relu')(dropout1)
    # dropout2=Dropout(rate=0.3)(dense2)
    # predictions=Dense(2, activation='sigmoid')(dropout2)

    flatten=Flatten()(resnet50.output)
    dense1=Dense(128, activation='relu')(flatten)
    dropout1=Dropout(rate=0.3)(dense1)
    dense2=Dense(64, activation='relu')(dropout1)
    dropout2=Dropout(rate=0.3)(dense2)
    predictions=Dense(2, activation='sigmoid')(dropout2)


    model = Model(inputs=resnet50.input, outputs=predictions)


    # compile the models
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # shuffle 
    indices = tf.range(start=0, limit=len(one_hots))
    idx = tf.random.shuffle(indices)
    images = tf.gather(images, idx)
    one_hots = tf.gather(one_hots, idx)

    # # train the models
    # history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    # print(images)
    # print(one_hots)
    print(images[:1904].shape)
    print(one_hots[:1904].shape)
    history = model.fit(images[:1904], one_hots[:1904], batch_size=128, epochs=3, 
                        validation_data=(images[1904:], one_hots[1904:]))



if __name__ == "__main__":
    preprocess_images()
    get_labels()
    images, one_hots = unpickle()
    train_model(images, one_hots)


