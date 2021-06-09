'''
Program to train a Convolution Neural Network on classification on
neurons cell types from digitally reconstructed images.

Authors: Isak Bengtson, Mattias Wedin

Sources:
https://www.tensorflow.org/tutorials/images/classification#create_a_dataset

'''
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib

# Neural network
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf

# SVM Classifier
from sklearn.metrics import confusion_matrix

import json


BATCH_SIZE = 16
IMG_HEIGHT = 300
IMG_WIDHT = 300
# Paths to image files
PATH_TO_DATA_UNBALANCED = "../data/images/mouse_unbalanced/"
PATH_TO_DATA_BALANCED = "../data/images/mouse_balanced/"

# Path to result unbalanced 
PATH_TO_RESULT_UB = '../data/result_after10_unbalanced.json'
PATH_TO_RESULT_ACC_UB = '../data/result_after10_accuracy_unbalanced'
PATH_TO_RESULT_IND_UB = '../data/result_after10_accuracy_individual_unbalanced.json'

# Path to result balanced
PATH_TO_RESULT_B = '../data/result_after10_balanced.json'
PATH_TO_RESULT_ACC_B = '../data/result_after10_accuracy_balanced'
PATH_TO_RESULT_IND_B = '../data/result_after10_accuracy_individual_unbalanced.json'

RUNS = 10

# Get the data from disc
def get_data(path_data):
    # path to image directory
    data_dir = pathlib.Path(path_data)

    # 70% of the data for training
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDHT),
        batch_size=BATCH_SIZE)

    # 30% of data for validation
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDHT),
        batch_size=BATCH_SIZE)
    return train_ds, val_ds


# Print some images for debugging
def print_images(train_ds):
    class_names = train_ds.class_names
    print(train_ds.class_names)

    # Print some images, just for debugging
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        
    plt.savefig(str(i) + ".png")

# Configurate the dataset
def config_dataset(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    print('Autotune', AUTOTUNE)
    # Use buffering prefetching
    train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def get_model(num_classes):
    # Create the model
    # num_classes = 3

    model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDHT, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # Dropout to prevent some overfitting
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # View model summary
    model.summary()
    return model

EPOCHS = 10

# Train the model
def train_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    return history, model


# Create plot of loss and accuracy on the traingin and validation sets
def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("../results/loss_and_acc_new.png")



# Run the classification N times

def run_classification(train_ds, val_ds, results, result_acc, result_individual, run):
    # Get the data
    #train_ds, val_ds = get_data()
    
    class_names = train_ds.class_names

    # configure the dataset
    train_ds, val_ds = config_dataset(train_ds, val_ds)

    # get the model
    model = get_model()

    # Compile the model
    model = compile_model(model)

    # Fit the model on the data
    history, model = train_model(model, train_ds, val_ds)

    # Predict mode on validation data
    y_pred  = model.predict(val_ds)
    
    predicted_categories = tf.argmax(y_pred, axis=1)

    true_categories = tf.concat([y for x, y in val_ds], axis=0)

    matrix = confusion_matrix(predicted_categories, true_categories)
    tot_correct = 0
    tot_incorrect = 0

    # Add result to correct dic
    for i in range(len(matrix)):
        tot  = sum(matrix[i])
        tot_correct += (matrix[i][i])
        tot_incorrect += (tot - matrix[i][i])
        results['CNN'][class_names[i]][0] += matrix[i][i]
        results['CNN'][class_names[i]][1] += tot - matrix[i][i]
        result_individual['CNN'][run][class_names[i]][0] = matrix[i][i]
        result_individual['CNN'][run][class_names[i]][1] = tot - matrix[i][i]


    acc = tot_correct / float(tot_incorrect + tot_correct)
    result_acc['CNN'].append(acc)
    
    return results, result_acc, result_individual

# def predict_data(model, class_names):
#     print("here we should try and do some prediction")
#     rat_path = pathlib.Path('../data/prediction_data/mouse_pred/mouse_A3-May14-IR1-5-G.png')
#     print(rat_path)
#     img = keras.preprocessing.image.load_img(
#         rat_path, target_size=(IMG_HEIGHT, IMG_WIDHT)
#     )
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0) # Create a batch

#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])

#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score))
#     )

import resource
  
def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


def run_test(runs, path_to_data, num_classes):
    train_ds, val_ds = get_data(path_to_data)

    # Create result dictionary
    print(train_ds.class_names)
    result = {}
    result['CNN'] = {}

    # Create accuracy result dic
    result_acc = {}
    result_acc['CNN'] = []

    # Create individual result dic
    result_individual = {}
    result_individual['CNN'] = {}

    for run in range(runs):
        result_individual['CNN'][run] = {}
        for name in train_ds.class_names:
            result_individual['CNN'][run][name] = [0,0]

    for name in train_ds.class_names:
        result['CNN'][name] = [0,0]
    
    for it in range(runs):
        result, result_acc, result_individual = run_classification(train_ds, val_ds, result, result_acc, result_individual, it, num_classes)
    
    return result, result_acc, result_individual


def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def run(path_to_data, path_tot, path_acc, path_ind, num_classes):
    f_tot = open(path_tot)
    f_acc = open(path_acc)
    f_individual = open(path_ind)
    cnn_data_tot, cnn_data_acc, cnn_data_individual  = run_test(10, path_to_data, num_classes)
    data_tot = json.load(f_tot)
    data_acc = json.load(f_acc)
    data_individual = json.load(f_individual)
    data_tot['CNN'] = cnn_data_tot['CNN']
    data_acc['CNN'] = cnn_data_acc['CNN']
    data_individual['CNN'] = cnn_data_individual['CNN']
    
    with open(path_tot, 'w', encoding='utf-8') as outfile:
        json.dump(data_tot, outfile, ensure_ascii=False, indent=4 ,default=convert) 

    with open(path_acc, 'w', encoding='utf-8') as outfile:
        json.dump(data_acc, outfile, ensure_ascii=False, indent=4 ,default=convert) 

    with open(path_ind, 'w', encoding='utf-8') as outfile:
        json.dump(data_individual, outfile, ensure_ascii=False, indent=4 ,default=convert) 

if __name__ == "__main__":
    balanced = False
    if len(sys.argv) > 1:
        if str(sys.argv[2]) == "balanced":
            balanced = True
    
    if balanced:
        run(PATH_TO_DATA_BALANCED, PATH_TO_RESULT_B, PATH_TO_RESULT_ACC_B, PATH_TO_RESULT_IND_B, 3)
    else:
        run(PATH_TO_DATA_UNBALANCED, PATH_TO_RESULT_UB, PATH_TO_RESULT_ACC_UB, PATH_TO_RESULT_IND_UB, 49)
    




