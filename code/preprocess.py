import pickle
import tensorflow as tf
import numpy as np

def get_labels():
    labels = []
    id_to_label = {}
    line_count = 0


    # The malignant or benign classifier is located at index 6 of the csv!
    # though, it can be 7 sometimes, depending on how many commmas a name has.
    with open('data/metadata', 'r') as file:
        for line in file:
            line_count += 1
            if line_count == 1:
                continue
            line_array = line.split(",")

            if 'benign' in line_array:
                id_to_label[line_array[0]] = 0
            elif 'malignant' in line_array:
                id_to_label[line_array[0]] = 1

            
    print(line_count)
    labels = list(id_to_label.values())


    ############################## PRINTING AID! ###################################
    # for key in id_to_label.keys():
    #   if id_to_label[key] != 'malignant' and id_to_label[key] != 'benign':
    #     print(f'this is the key {key} and value {id_to_label[key]}')
    ################################################################################
    print(f'Labels: {labels}')
    print(f'Total labels count: {len(labels)}')
    print(f'Benign: {labels.count(0)}')
    print(f'Malignant: {labels.count(1)}')

def unpickle():
    data = {}
    with open('data/data-001.p', 'rb') as file:
        data = pickle.load(file)

    images = np.array([data[key][0] for key in data.keys()])
    indices = [data[key][1] for key in data.keys()]
    one_hots = tf.one_hot(indices, 2)

    print(images.shape)
    print(one_hots.shape)


if __name__ == "__main__":
    get_labels()
    unpickle()

