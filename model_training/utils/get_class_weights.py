from path import Path
import numpy as np
from skimage.io import imread


def calculate_class_weights_from_files(label_files, num_classes=4):
    """
    Calculate class weights based on the inverse frequency of pixels in each class from a list of label files.
    :param label_files: List of file paths to label images (each file contains class labels for segmentation).
    :param num_classes: Number of classes in the segmentation task.
    :return: Normalized class weights for CrossEntropyLoss-like function.
    """
    pixel_counts = np.zeros(num_classes)  # Initialize pixel counts for each class

    # Loop through each label file to count the number of pixels per class
    for label_file in label_files:
        # Assuming label_file contains the path to a NumPy array with class labels
        labels = imread(label_file)  # Replace this with appropriate loading method (e.g., from PNG, TIFF, etc.)

        # Check if the label file contains class labels and process them
        if labels.shape:  # Ensure file is not empty
            for class_id in range(num_classes):
                pixel_counts[class_id] += np.sum(labels == class_id)  # Count pixels for each class

    # Calculate class frequencies
    class_frequencies = pixel_counts / np.sum(pixel_counts)

    # Calculate inverse frequencies for class weights
    class_weights = 1.0 / (class_frequencies + 1e-6)  # Add small value to avoid division by zero

    # Normalize the weights (optional but recommended)
    class_weights = class_weights / np.sum(class_weights)

    return class_weights


if __name__ == '__main__':
    # Define the path to the label files
    # Define the path to the label files
    data_folder = Path(r'F:\Neavus\resized_export_512')

    label_files = data_folder.glob('*/*corrected.tiff')

    # Calculate class weights from the label files
    class_weights = calculate_class_weights_from_files(label_files, num_classes=4)

    print(f'Class weights: {class_weights}')

