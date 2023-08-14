# image_path = 'pathtotheimage'

import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def list_images(directory_path):
    # List files in the directory
    file_list = os.listdir(directory_path)

    # Display the list of files in the directory
    print("Files in the directory:")
    for index, file in enumerate(file_list):
        print(f"{index + 1}. {file}")

    return file_list


def convert_tiff_to_numpy(image_path):
    im = io.imread(image_path)
    # reorder axes to be: x,y,z,ch
    im.transpose(1, 2, 0, 3)

    return im


def main():
    # Get directory path from user input
    # directory_path = input("Enter the path to the directory containing your images: ")
    directory_path = "/Users/yoadoxman/Documents/ Weizmann/ML Course 2023/Project"

    # List images and get the selected file
    file_list = list_images(directory_path)
    file_index = int(input("Enter the number of the image file you want to process: ")) - 1
    selected_file = file_list[file_index]

    # Construct the full path to the selected image
    image_path = os.path.join(directory_path, selected_file)

    image_np = convert_tiff_to_numpy(image_path)

    plt.imshow(image_np, cmap='gray')
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()


if __name__ == "__main__":
    main()
