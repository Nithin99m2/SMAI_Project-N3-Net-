from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


import shutil
import os
import numpy as np
import argparse


def get_files_from_folder(path):
    # to get the files from the folder we use this function
    files = os.listdir(path)
    return np.asarray(files)


def main_func(path_to_data, path_to_test_data, train_ratio):
    # here we get the directories to load the data
    _, dirs, _ = next(os.walk(path_to_data))

    # Here we are calculating the data for each of the class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # Transfering our files to the dataframe
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])

        # creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("traindatapath: ", required=True)
    parser.add_argument("testdatapath", required=True)
    parser.add_argument("traintest_split", required=True)
    return parser.parse_args()


def descreteIntensity_scale(df, f, T):
    temp2 = T*df[np.random.randint(len(f))]
    return temp2


def continuousIntensity_scale(df, f, T):
    temp2 = T*df[np.random.randint[1](len(f))]
    return temp2


if __name__ == "__main__":
    args = parse_args()
    main_func(args.data_path, args.test_data_path_to_save,
              float(args.train_ratio))

    (X_train, Y_train), (X_test, Y_test) = os.path('./dst')

    images = X_train.reshape(10000, 28, 28, 1).astype('float64')

    print(images.shape)
    print(type(images))
    print(images.size)

    # just checking the features of 1 image
    img = load_img('./temp.jpg', grayscale=True)

    # image to numpy array
    img_array = img_to_array(img)

    # saving the image
    save_img('grayscale.jpg', img_array)

    # loading the image
    img = load_img('grayscale.jpg')
    print(type(img))
    print(img.format)
    print(img.mode)
    print(img.size)

    image_vec_train = []

    image_vec_test = []

    for i in range(X_train):
        te1 = img_to_array(i)
        image_vec_train[i] += [te1]

    for i in range(X_test):
        te1 = img_to_array(i)
        image_vec_test[i] += [te1]

    print(len(image_vec_test))

    print(len(image_vec_train))

    image_details_train = []
    image_details2_test = []

    for i in range(len(image_vec_train)):
        s1 = type(image_vec_train[i])
        s2 = image_vec_train[i].format
        s3 = image_vec_train[i].mode
        s4 = image_vec_train[i].size

        image_details_train.append([s1, s2, s3, s4])

    for i in range(len(image_vec_test)):
        s1 = type(image_vec_test[i])
        s2 = image_vec_test[i].format
        s3 = image_vec_test[i].mode
        s4 = image_vec_test[i].size

        image_details2_test.append([s1, s2, s3, s4])

    print(image_details_train)
    print(image_details2_test)

    print(image_vec_test)
