# detection/sliding_window.py
# This module provides the sliding window functionality for object detection.

def sliding_window(image, step_size, window_size):
    """
    Yields windows from an image to be used for detection.

    This function slides a window across the image, yielding each window's
    coordinates and the patch of the image it contains.

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
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Yield the current window's coordinates and image patch
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])