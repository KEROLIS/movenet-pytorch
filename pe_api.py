import cv2
import torch
from fastapi import FastAPI, UploadFile
from movenet.models.model_factory import load_model
from movenet.utils import _process_input, draw_skel_and_kp
import numpy as np
import base64
app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("movenet_lightning", ft_size=48).to(device)

@app.post("/draw_keypoints")
async def draw_keypoints(file: UploadFile, conf_thres: float = 0.3):
    # Read the image
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    input_image, draw_image = _process_input(image, 192)
    
    # Predict the keypoints
    with torch.no_grad():
        input_image = torch.Tensor(input_image).to(device)
        kpt_with_conf = model(input_image)[0, 0, :, :]
        kpt_with_conf = kpt_with_conf.cpu().numpy()

    # Draw the keypoints on the image
    draw_image = draw_skel_and_kp(draw_image, kpt_with_conf, conf_thres=conf_thres)


    return  {"image": base64.b64encode(cv2.imencode('.jpg', draw_image)[1]).decode('utf-8')}