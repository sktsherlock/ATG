import os
import shutil
import argparse

def main(dataset_folder, train_ratio, val_ratio, test_ratio):
    train_folder = os.path.join(dataset_folder, 'train')
    val_folder = os.path.join(dataset_folder, 'val')
    test_folder = os.path.join(dataset_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for class_folder in os.listdir(dataset_folder):
        if class_folder not in ['train', 'val', 'test']:
            class_path = os.path.join(dataset_folder, class_folder)
            train_class_path = os.path.join(train_folder, class_folder)
            val_class_path = os.path.join(val_folder, class_folder)
            test_class_path = os.path.join(test_folder, class_folder)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(val_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)
            image_files = os.listdir(class_path)
            num_train = int(len(image_files) * train_ratio)
            num_val = int(len(image_files) * val_ratio)
            train_images = image_files[:num_train]
            val_images = image_files[num_train:num_train + num_val]
            test_images = image_files[num_train + num_val:]
            for image_file in train_images:
                src_file = os.path.join(class_path, image_file)
                dest_file = os.path.join(train_class_path, image_file)
                shutil.move(src_file, dest_file)
            for image_file in val_images:
                src_file = os.path.join(class_path, image_file)
                dest_file = os.path.join(val_class_path, image_file)
                shutil.move(src_file, dest_file)
            for image_file in test_images:
                src_file = os.path.join(class_path, image_file)
                dest_file = os.path.join(test_class_path, image_file)
                shutil.move(src_file, dest_file)

    def rename_files(folder_path, prefix):
        for class_folder in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_folder)
            for i, image_file in enumerate(os.listdir(class_path)):
                ext = os.path.splitext(image_file)[1]
                new_name = f'{prefix}_{i}{ext}'
                old_file = os.path.join(class_path, image_file)
                new_file = os.path.join(class_path, new_name)
                os.rename(old_file, new_file)

    # rename_files(train_folder, 'train')
    # rename_files(val_folder, 'val')
    # rename_files(test_folder, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for dataset organization')
    parser.add_argument('--dataset_folder', type=str, help='Path to the dataset folder')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of testing data')
    args = parser.parse_args()

    main(args.dataset_folder, args.train_ratio, args.val_ratio, args.test_ratio)