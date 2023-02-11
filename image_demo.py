import cv2
import torch

# Import load_model function and read_imgfile and draw_skel_and_kp functions from movenet library
from movenet.models.model_factory import load_model
from movenet.utils import read_imgfile, draw_skel_and_kp


def draw_keypoints_on_image(model_name, image, conf_threshold=0.3):
    # Load the specified model
    model = load_model(model_name, ft_size=48)

    # Read the input image and prepare it for processing
    input_image, draw_image = read_imgfile(
        image, 192)
    with torch.no_grad():
        # Convert the input image to a Tensor
        input_image = torch.Tensor(input_image) 

        # Run the model on the input image and get the keypoints with confidence
        kpt_with_conf = model(input_image)[0, 0, :, :]
        kpt_with_conf = kpt_with_conf.numpy()

    # Draw the skeleton and keypoints on the image
    draw_image = draw_skel_and_kp(
        draw_image, kpt_with_conf, conf_thres=conf_threshold)
    
    # Return the image with the keypoints drawn on it
    return draw_image
