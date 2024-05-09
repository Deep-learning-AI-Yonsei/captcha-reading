import os
import shutil
from sklearn.model_selection import train_test_split

# Utility function to create directory if not exists
def create_directory(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        file_names.append(filename)
    return file_names

# Process and split dataset
def split_dataset(rawdata_dir, output_dir = "./dataset/"):
    if os.path.exists(output_dir): return 0

    images = get_file_names(rawdata_dir)
    
    # Split the images into sets
    train_images, temp_images = train_test_split(images, train_size=0.8)
    val_images, test_images = train_test_split(temp_images, train_size=0.5)

    # Define function for saving and grayscaling images
    create_directory(output_dir)
    def process_images(images, data_type):
        dir_path = os.path.join(output_dir, data_type)
        create_directory(dir_path)

        for image in images:
            shutil.copy(os.path.join(rawdata_dir, image), os.path.join(dir_path, image))

    # Process each set
    process_images(train_images, 'train')
    process_images(val_images, 'val')
    process_images(test_images, 'test')

    return 1