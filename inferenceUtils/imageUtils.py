import numpy as np
import cv2

def displayTorchImg(torchImage, name: str, persistImages = False):
    '''take torch tensor input of shape [1, 256, 256], convert to numpy, rescale and display'''
    npImage = torchInputToNumpy(torchImage)
    display_depth_image_with_colormap(npImage, name, persistImages)

def torchInputToNumpy(torchDepthImg):
    '''convert torch image to numpy and display'''
    tensor_image = torchDepthImg.squeeze(0) # remove batch dim to get size [256, 256]
    numpy_image = tensor_image.numpy() # convert to numpy

    # rescale for numpy display and change to uint8
    min_val, max_val = numpy_image.min(), numpy_image.max()
    if max_val > 255 or min_val < 0:
        numpy_image = (numpy_image - min_val) / (max_val - min_val) * 255  # Scale to [0, 255]
        numpy_image = numpy_image.astype(np.uint8)  # Convert to uint8 for display
    return numpy_image

def display_depth_image_with_colormap(depth_image, name: str, persistImages = False):
    '''display numpy image with opencv and high-contrast colour map'''
    # Check if the image needs to be scaled to 0-255 range
    if depth_image.max() > 255 or depth_image.min() < 0:
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

    depth_image = depth_image.astype(np.uint8) # Ensure the image is in uint8 format for proper display
    color_mapped_img = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET) # Apply the color map
    cv2.imshow('Depth Image with COLORMAP_JET - '+name, color_mapped_img) # Display the image using OpenCV
    
    if (persistImages == False):
        cv2.waitKey(0) # Wait for a key press and close the window
        cv2.destroyAllWindows()