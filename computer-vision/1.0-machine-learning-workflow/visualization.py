from utils import get_data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_image_with_bboxes(image_object):


    # Load the image
    image_path = 'data/images/'+image_object['filename']
    bboxes = image_object['boxes']
    classes = image_object['classes']

    image = Image.open(image_path)

    # Create a figure and axes
    _, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    for bbox in bboxes:
        # Create a rectangle patch
        rectangle = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axes
        ax.add_patch(rectangle)

    # Show the image with the bounding box
    plt.show()

def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """

    for image_object in ground_truth:
        plot_image_with_bboxes(image_object)



if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)