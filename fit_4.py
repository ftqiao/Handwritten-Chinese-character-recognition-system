import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import os


# Function to generate log directory
def get_run_logdir():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(os.curdir, "logs", run_id)


# Function to load the data as a generator using tf.data API
def load_data(batch_size, train_dir, test_dir):
    '''Load the data as a generator'''
    # Define data augmentation techniques
    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    # Define training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(64, 64),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    # Define testing dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(64, 64),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    return train_dataset, test_dataset


# Load the data
batch_size = 16
train_dataset, test_dataset = load_data(batch_size, 'data/train', 'data/test')
# Define the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Define callbacks
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=get_run_logdir())
checkpoint_callback = keras.callbacks.ModelCheckpoint('model/best_model.h5',
                                                      monitor='val_accuracy',
                                                      save_best_only=True)
# Train the model using generator
model.fit(train_dataset,
          epochs=80,
          validation_data=test_dataset,
          callbacks=[tensorboard_callback, checkpoint_callback])
