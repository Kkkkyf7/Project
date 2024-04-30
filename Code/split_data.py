import os
from shutil import copy, rmtree
import random

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # if file exists, delete the original file and creat a new one.
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    # Random reproducibility is guaranteed
    random.seed(0)

    # 10% of the data in the dataset was divided into the validation set
    split_rate = 0.1

    data_root = "D:\\model"
    origin_disease_path = os.path.join(data_root, "dataset")
    assert os.path.exists(origin_disease_path), "path '{}' does not exist.".format(origin_disease_path)

    disease_class = [cla for cla in os.listdir(origin_disease_path)
                    if os.path.isdir(os.path.join(origin_disease_path, cla))]

    # Make a folder for the training set
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in disease_class:
        # Create folders for each category
        mk_file(os.path.join(train_root, cla))

    # Make a folder for the validation set
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in disease_class:
        # Create folders for each category
        mk_file(os.path.join(val_root, cla))

    for cla in disease_class:
        cla_path = os.path.join(origin_disease_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # The index of the randomly sampled validation set
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # Copy the files assigned to the validation set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # Copy the files assigned to the training set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
