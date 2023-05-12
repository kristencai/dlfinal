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

    # NOTE: loading in VGG when testing its performance 
    # vgg16 = tf.keras.applications.VGG16(include_top=False, weights = 'imagenet', input_shape = (256,256,3))

    # ======================================================================
    # ======================================================================

    # This section loads all of the data into a datset to ensure that the gpu doesn't 
    # overload 
    dataset = tf.data.Dataset.from_tensor_slices((images, one_hots))
    batch_size = 32
    # for invert, use 7100
    train_dataset = dataset.take(6000).shuffle(buffer_size=6000)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = dataset.skip(6000).batch(batch_size)

    # ======================================================================
    # ======================================================================

    # NOTE: This is a failed preprocessing technique: 
    # # PREPROCESSING WITH VGG 
    # vgg_model = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', input_shape=(256,256,3))

    # slice_model = Model(inputs = vgg_model.input, outputs = vgg_model.get_layer('block1_conv1').output)

    # preprocessed = slice_model.predict(images, batch_size = 16)

    # print('preprocessed shape: {preprocessed.shape}')

    # ABSTRACTING WITH RESNET50

    # ======================================================================
    # ======================================================================

    resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(256,256,3))

    for layer in resnet50.layers:
        layer.trainable = False

    flatten=Flatten()(resnet50.output)
    dense1=Dense(512, activation='relu')(flatten)
    print(f'dense 1: {dense1.shape}')
    dropout1=Dropout(rate=0.3)(dense1)
    dense2=Dense(256, activation='relu')(dropout1)
    print(f'dense 2: {dense2.shape}')
    dropout2=Dropout(rate=0.5)(dense2)
    dense3 = Dense(64, activation = 'relu')(dropout2)
    print(f'dense 3: {dense3.shape}')
    predictions=Dense(2, activation='softmax')(dense3)
    print(f'dense 4: {predictions.shape}')

    model = Model(inputs=resnet50.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0004), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)

    # ======================================================================
    # ======================================================================

    # NOTE: The below is manual shuffling without using datasets. 

    # indices = tf.range(start=0, limit=len(one_hots))
    # idx = tf.random.shuffle(indices)
    # images = tf.gather(images, idx)
    # one_hots = tf.gather(one_hots, idx)
    # history = model.fit(images[:7100], one_hots[:7100], batch_size=64, epochs=5, 
    #                     validation_data=(images[7100:], one_hots[7100:]))

    # compile the models
    # history = model.fit(images[:1500], one_hots[:1500], batch_size=64, epochs=5, 
    #                     validation_data=(images[1500:], one_hots[1500:]))
    
    # ======================================================================
    # ======================================================================

    # NOTE: This was a manual implementaion of fit, because I thought there was an issue
    #  with the fit function, but it was really a cuda issue that i could not fix.
    # num_epochs = 5
    # for epoch in range(num_epochs):
    #     batch_num = 0
    #     print(f'Epoch {epoch}: ------')
    #     for x_batch, y_batch in train_dataset:
    #         print(f'Batch{batch_num} ------')
    #         with tf.GradientTape() as tape:
    #             y_pred = model(x_batch)
    #             # print(f'shape of x_batch {x_batch.shape}')
    #             # print(f'shape of y_batch {y_batch.shape}')
    #             loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #         # Evaluate on validation data
    #     total_loss = 0
    #     total_accuracy = 0
    #     num_batches = 0
    #     for x_val_batch, y_val_batch in val_dataset:
    #         predictions = model(x_val_batch)
    #         loss = tf.keras.losses.categorical_crossentropy(y_val_batch, predictions)
    #         accuracy = tf.keras.metrics.categorical_accuracy(y_val_batch, predictions)
    #         total_loss += tf.reduce_mean(loss)
    #         total_accuracy += tf.reduce_mean(accuracy)
    #         num_batches += 1
    #     val_loss = total_loss / num_batches
    #     val_accuracy = total_accuracy / num_batches
    #     # Print results
    #     print(f'Epoch {epoch + 1}/{num_epochs}, loss: {val_loss:.3f}, accuracy: {val_accuracy:.3f}')

    # ======================================================================
    # ======================================================================

    



if __name__ == "__main__":
    # preprocess_images()
    # get_labels()
    # images, one_hots = unpickle_vgg()
    images, one_hots = unpickle()
    train_model(images, one_hots)


