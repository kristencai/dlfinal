import pickle
import tensorflow as tf
import numpy as np
# import Image
from PIL import Image
from tqdm import tqdm
import os

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
    # print(f'Labels: {labels}')
    print(f'Total labels count: {len(labels)}')
    print(f'Benign: {labels.count(0)}')
    print(f'Malignant: {labels.count(1)}')

def preprocess_images():
    #trying more preprocessing stuff:
    directory = 'images'

    # picture_id -> (pixel values, label value)
    data = {}
    count = 0
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
    # this is the final code bc it resizes all of them
    # for each photo, we resize, get the pixel values, and reshape to (256, 256, 3)
    for filename in tqdm(os.scandir(directory)):
        # count += 1
        # if count > 100:
        #   break
        if filename.is_file():
            with Image.open(filename.path) as img:
              # print(filename.path.strip('images/').strip('.JPG'))
              if filename.path.strip('images/').strip('.JPG') in id_to_label:
                new_image = img.resize((256,256))
                # new_image.show()

                ninety = new_image.rotate(90.0)
                # ninety.show()
                ninety = np.array(list(ninety.getdata()))
                ninety = np.reshape(ninety, (256, 256, 3))

                one_eighty = new_image.rotate(180.0)
                # one_eighty.show()
                one_eighty = np.array(list(one_eighty.getdata()))
                one_eighty = np.reshape(one_eighty, (256, 256, 3))


                two_seventy = new_image.rotate(270.0)
                # two_seventy.show()
                two_seventy = np.array(list(two_seventy.getdata()))
                two_seventy = np.reshape(two_seventy, (256, 256, 3))

                new_image = np.array(list(new_image.getdata()))
                new_image = np.reshape(new_image, (256, 256, 3))
                data[filename.path.strip('images/').strip('.JPG')] = \
                  (new_image , ninety, one_eighty, two_seventy, id_to_label[filename.path.strip('images/').strip('.JPG')])

              # else:
                # print(filename.path.strip('images/').strip('.JPG'))

    # print(f'total pictures looped through: {count}')
    print(f'data length : {len(data)} ')

    # one hotting all of the returned indeces
    ################################################################################
    ################################################################################
    ################################################################################
    # This section of the cell will pickle the above preprocessed pictures.
    # By doing this, we only have to run this cell ONCE (on kristen's computer
    # cuz she has the vim)

    with open(f'data/data.p', 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    print(f'Data has been dumped into data.p!')

    # maybe consider using PIL.Image.radial_gradient(mode) to reduce the noise on
    # on the peripheries of the image.

def unpickle():
    data = {}
    with open('data/data.p', 'rb') as file:
        data = pickle.load(file)

    images = np.array([data[key][0] for key in data.keys()])
    indices = [data[key][1] for key in data.keys()]
    one_hots = tf.one_hot(indices, 2)

    print(images.shape)
    print(one_hots.shape)

    return images, one_hots
