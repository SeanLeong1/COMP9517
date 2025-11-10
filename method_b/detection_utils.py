import cv2

def image_pyramid(image, scale=1.5, min_size=(100, 100)):
    """
    Generates an image pyramid (multi-scale representations) for the input image.
    This allows detecting objects at different sizes.
    [cite from user prompt]
    """
    
    # Yield the original image
    yield image

    while True:
        # Compute the new dimensions of the image based on the scale factor
        new_width = int(image.shape[1] / scale)
        new_height = int(image.shape[0] / scale)
        
        # Resize the image
        # We use cv2.INTER_AREA for shrinking, as it gives good results
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # If the new image is smaller than the minimum size, stop the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # Yield the new (smaller) image
        yield image

def sliding_window(image, window_size, step_size=32):
    """
    Slides a window across an image from left-to-right, top-to-bottom.
    [cite from user prompt]
    """
    
    # Get the dimensions of the image
    (img_height, img_width) = image.shape[:2]
    (win_width, win_height) = window_size

    # Iterate over the y-coordinates (rows)
    for y in range(0, img_height - win_height + 1, step_size):
        # Iterate over the x-coordinates (columns)
        for x in range(0, img_width - win_width + 1, step_size):
            # Yield the current window's coordinates and the image patch
            # The patch is the sub-image defined by the window
            # The coordinates (x, y) are the top-left corner of the window
            patch = image[y:y + win_height, x:x + win_width]
            yield (x, y, patch)