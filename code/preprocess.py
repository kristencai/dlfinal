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
                                                                                                                                                                                   105,45        82%