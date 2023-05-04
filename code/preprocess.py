import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
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


    directory = 'images'

    # TODO: Try using a conv2d layer with multiple output channels to extract the 
    # rough features of the image and use that as input as well


    # picture_id -> (pixel values, label value)
    data = {}
    count = 0
    malignant = 0
    benign = 0
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
                    if id_to_label[filename.path.strip('images/').strip('.JPG')] == 1:
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

                        invert = ImageOps.invert(new_image)
                        invert = np.array(list(invert.getdata()))
                        invert = np.reshape(invert, (256, 256, 3))

                        new_image = np.array(list(new_image.getdata()))
                        new_image = np.reshape(new_image, (256, 256, 3))
                        data[filename.path.strip('images/').strip('.JPG')] = \
                        ([new_image , ninety, one_eighty, two_seventy, invert], id_to_label[filename.path.strip('images/').strip('.JPG')])
                        malignant += 4
                    else: 
                        new_image = img.resize((256,256))

                        one_eighty = new_image.rotate(180.0)
                        # one_eighty.show()
                        one_eighty = np.array(list(one_eighty.getdata()))
                        one_eighty = np.reshape(one_eighty, (256, 256, 3))

                        invert = ImageOps.invert(new_image)
                        invert = np.array(list(invert.getdata()))
                        invert = np.reshape(invert, (256, 256, 3))

                        new_image = np.array(list(new_image.getdata()))
                        new_image = np.reshape(new_image, (256, 256, 3))
                        data[filename.path.strip('images/').strip('.JPG')] = \
                        ([new_image, one_eighty, invert] , id_to_label[filename.path.strip('images/').strip('.JPG')])
                        benign += 2

    print(f' overall malignant: {malignant} v. benign: {benign}')

    # change the file path depending on which version of preprocessing
    with open(f'data/invert.npy', 'wb') as pickle_file:
        np.save( pickle_file, data)
    print(f'Data has been dumped into data.npy!')

# def preprocess_images():
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


    vggModel = tf.keras.applications.vgg16.VGG16(
        include_top=True, 
        weights='imagenet', 
        input_tensor=None, 
        input_shape= None, 
        pooling=None, 
        classes=1000)
    
    directory = 'images'

    # TODO: Try using a conv2d layer with multiple output channels to extract the 
    # rough features of the image and use that as input as well
    

    # picture_id -> (pixel values, label value)
    data = {}
    count = 0
    malignant = 0
    benign = 0
    # this is the final code bc it resizes all of them
    # for each photo, we resize, get the pixel values, and reshape to (256, 256, 3)
    for filename in tqdm(os.scandir(directory)):
        count += 1
        if count > 1750:
          break
        count += 1
        print(f'on iteration: {count}')
        if filename.is_file():
            with Image.open(filename.path) as img:
            # print(filename.path.strip('images/').strip('.JPG'))
                if filename.path.strip('images/').strip('.JPG') in id_to_label:
                    new_image = img.resize((224,224))
                    # new_image.show()
                    
                    new_image = np.array(list(new_image.getdata()))
                    new_image = np.reshape(new_image, (224, 224, 3))
                    slice_model  = tf.keras.Model(inputs=vggModel.input, outputs= vggModel.get_layer('block1_conv1').output)
                    slice_output = slice_model.predict(new_image[None,:,:,:])
                    slice_output = tf.reshape(slice_output, (224, 224, 64))
                    data[filename.path.strip('images/').strip('.JPG')] = (slice_output, id_to_label[filename.path.strip('images/').strip('.JPG')])



    # print(f' overall malignant: {malignant} v. benign: {benign}') 

    with open(f'data/vggpreprocess.npy', 'wb') as pickle_file:
        np.save( pickle_file, data)
    print(f'Data has been dumped into data.npy!')



# def unpickle_vgg():
    data = {
    }
    with open('data/vggpreprocess.npy', 'rb') as file:
        data= np.load(file, allow_pickle=True).item()
    print(f'data successfully unpickled')
    images = np.array([[data[key][0]] for key in data.keys()])
    one_hots = tf.one_hot([data[key][1] for key in data.keys()], 2)

    return images, one_hots



def unpickle():
    data = {}
    # change the file path dependingo n which version of preprocessing
    with open('data/data.npy', 'rb') as file:
        data = np.load(file, allow_pickle=True).item()
    print(f'data successfully unpickled')

    print(f'this is th elen of the data:{len(data)}')
    pictures = []
    indices = []
    for key in data.keys():
        for i in data[key][0]:
            pictures.append(i)
            indices.append(data[key][1])

    images = np.array(pictures)
    # indices = [data[key][1] for key in data.keys()]
    one_hots = tf.one_hot(indices, 2)

    print(images.shape)
    print(one_hots.shape)

    return images, one_hots




