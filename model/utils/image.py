import configparser
import imageio
import json
import numpy as np
from PIL import Image, ImageOps


conf = configparser.ConfigParser()
conf.read('config.ini')

CONFIG = conf['vgg-19']


def generate_noise_image(content_image, noise_ratio):
    """Generate a noisy image by adding random noise to the content_image.

    Args:
    content_image: Numpy array of input content image
    noise_ratio: float, between (0,1) exclusive, ratio of noise to add to image

    Returns:
    input_image: Numpy array of image with noise added

    """   
    # Generate a random noise_image
    noise_image = np.random.uniform(
        -20,
        20,
        (
            1,
            int(CONFIG['IMAGE_HEIGHT']),
            int(CONFIG['IMAGE_WIDTH']),
            int(CONFIG['COLOR_CHANNELS']),
        )).astype('float32')
    
    # Set to be a weighted average of the content_image and noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image

def resize_and_crop_image(image,
                          size=(int(CONFIG['IMAGE_WIDTH']),
                                int(CONFIG['IMAGE_HEIGHT']))):
    """Resize and crop image to size.

    Args:
    image: Image file object
    size: (width, height) tuple, output image size

    Returns:
    image: Resized numpy array image
    """
    if image.size != size:
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image = np.array(image)

    return image


def reshape_and_normalize_image(image):
    """Reshape and normalize image.

    Normalize using means from config file.

    Args:
    image: Numpy array image

    Returns:
    image: Resized and normalized image numpy array
    """
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    means = np.array(json.loads(CONFIG['MEANS'])).reshape((1,1,1,3))
    
    # Substract the mean to match the expected input of VGG16
    image = image - means
    
    return image


def save_image(path, image):
    """Un-normalize and save image.

    Un-normalize using means from config file.

    Args:
    path: str file path for saved image
    image: Numpy array image

    Returns:
    None
    """
    means = np.array(json.loads(CONFIG['MEANS'])).reshape((1,1,1,3))
    # Un-normalize the image so that it looks good
    image = image + means
    
    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imsave(path, image)