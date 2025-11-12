# detection/sliding_window.py
# This module provides the sliding window and image pyramid functionality for object detection.

import cv2

def pyramid(image, scale=1.5, min_size=(30, 30)):
    """
    Yields successive layers of an image pyramid.

    Args:
        image (np.ndarray): The image to create a pyramid from.
        scale (float): The factor by which to scale down the image at each layer.
        min_size (tuple): The minimum (width, height) of a layer. Once a layer
                          is smaller than this, the pyramid construction stops.

    Yields:
        np.ndarray: The next layer in the pyramid.
    """
    # Yield the original image
    yield image

    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        
        # Stop if the new dimensions are too small
        if w < min_size[0] or h < min_size[1]:
            break
            
        image = cv2.resize(image, (w, h))

        # Yield the next image in the pyramid
        yield image

def sliding_window(image, step_size, window_size):
    """
    Yields windows from an image to be used for detection.

    This function slides a window across the image, yielding each window's
    coordinates and the patch of the image it contains. This is a generator
    function, which is more memory-efficient.

    Args:
        image (np.ndarray): The image to slide over.
        step_size (int): The number of pixels to "step" in each direction.
        window_size (tuple): The (width, height) of the window.

    Yields:
        tuple: A tuple containing (x, y, window), where (x, y) are the
               top-left coordinates of the window, and window is the
               image patch itself.
    """
    # Slide the window across the image
    # Note: We subtract window_size to prevent the window from going off-image.
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Yield the current window's coordinates and image patch
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])