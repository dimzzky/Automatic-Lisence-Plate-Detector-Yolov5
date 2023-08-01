import torch
from pathlib import Path
import cv2
import numpy as np

from models.yolo import Model
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the model
weights_path = Path('models/yolov5s_custom.pt')  # replace with your path

# Load model, this will load architecture and weights
device = select_device('cpu')
model = attempt_load(weights_path)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

# Load the image
image_path = 'test_images/New_Indonesian_License_Plate_for_Cars.jpg'  # replace with your path
img0 = cv2.imread(image_path)  # BGR
img = img0.copy()

# Padded resize
img = cv2.resize(img, (imgsz, imgsz))
# Normalize RGB
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
img = np.ascontiguousarray(img)

img = torch.from_numpy(img).to(device)
img = img.float()  # uint8 to fp32
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# Inference
pred = model(img, augment=False)[0]

# Apply NMS
pred = non_max_suppression(pred, 0.75, 0.45, classes=None, agnostic=False)

# Process detections
for i, det in enumerate(pred):  # detections per image
    if len(det):
        # Rescale boxes from imgsz to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results to the screen
        for *xyxy, conf, cls in reversed(det):
            print(f'{cls}: {conf}')  # print class and confidence

            # Get bounding box coordinates
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            
            # Crop the image using the coordinates
            cropped_img = img0[y1:y2, x1:x2]

            # Save the cropped image (optional)
            output_path = 'output/roi.jpg'
            cv2.imwrite(output_path, cropped_img)