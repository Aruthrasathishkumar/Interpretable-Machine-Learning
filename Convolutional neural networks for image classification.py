from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_files(files_dir, img_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        files_dir,
        image_size=(img_size, img_size),
        batch_size=None,
        label_mode="int"
    )
    images, labels = [], []
    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

# # Dataset 1: Visual Domain Decathlon

class_names = ['0001', '0002', '0003', '0004', '0005',
               '0006', '0007', '0008', '0009', '0010']
img_size = 32
num_epochs = 15
batch_size = 160

train_path = r"D:\aruth\Documents\AIT 636 ML\Project_13\Visual Domain Decathlon (Subset)\train"
test_path = r"D:\aruth\Documents\AIT 636 ML\Project_13\Visual Domain Decathlon (Subset)\test"

for j in [1, 2, 3]:
    for k in [32, 64, 112]:
        img_size = k

        print("Dataset: Visual Domain Decathlon")
        print("Number of convolutional layers =", j, "and img_size =", img_size)

        train_images, train_labels = load_files(train_path, img_size)
        test_images, test_labels = load_files(test_path, img_size)

        train_images, test_images = train_images / 255.0, test_images / 255.0

        # plt.figure(figsize=(10, 10))
        # for i in range(25):
        #     plt.subplot(5, 5, i+1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.grid(False)
        #     plt.imshow(train_images[i])
        #     plt.xlabel(class_names[train_labels[i]])
        # plt.show()

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='relu',
                                      input_shape=(img_size, img_size, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        if j > 1:
            model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        if j > 2:
            model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.summary()

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=10))

        model.summary()

        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(x=train_images, y=train_labels, batch_size=batch_size,
                            epochs=num_epochs,
                            validation_data=(test_images, test_labels),
                            verbose=1)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('Test accuracy is: ', test_acc)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.title(f'Visual Domain Decathlon: Layers = {j}, Image Size = {img_size}')
        plt.show()


# Dataset 2: Cat-Dog
class_names = ['cat', 'dog']
img_size = 32
num_epochs = 15
batch_size = 160

train_dir = r"D:\aruth\Documents\AIT 636 ML\Project_13\Cat-Dog\train"
test_dir = r"D:\aruth\Documents\AIT 636 ML\Project_13\Cat-Dog\test"

for j in [1, 2, 3]:
    for k in [32, 64, 112]:
        img_size = k

        print("Dataset: Cat-Dog")
        print("Number of convolutional layers =", j, "and img_size =", img_size)

        train_images, train_labels = load_files(train_dir, img_size)
        test_images, test_labels = load_files(test_dir, img_size)

        train_images, test_images = train_images / 255.0, test_images / 255.0

        # plt.figure(figsize=(10, 10))
        # for i in range(25):
        #     plt.subplot(5, 5, i+1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.grid(False)
        #     plt.imshow(train_images[i])
        #     plt.xlabel(class_names[train_labels[i]])
        # plt.show()

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='relu',
                                      input_shape=(img_size, img_size, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        if j > 1:
            model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        if j > 2:
            model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.summary()

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=2))

        model.summary()

        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(x=train_images, y=train_labels, batch_size=batch_size,
                            epochs=num_epochs,
                            validation_data=(test_images, test_labels),
                            verbose=1)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('Test accuracy is: ', test_acc)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.title(f'Cat-Dog: Layers = {j}, Image Size = {img_size}')
        plt.show()