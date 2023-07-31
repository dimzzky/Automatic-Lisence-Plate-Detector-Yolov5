import torch
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')

device = torch.device('cpu')

from models.experimental import attempt_load

# Load the YOLOv5 model
weights_path = 'models/yolov5s_custom.pt'  # Path to the YOLOv5 weights file in the workspace
model = attempt_load(weights_path)
model.to(device).eval()

# Set the confidence threshold for detection
confidence_threshold = 0.75

# Load the input image
image_path = 'test_images/IMG_COM_20220219_2231341.jpg'  # Replace with the path to your image

try:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to PIL Image format
    image_pil = Image.fromarray(image_rgb)

    # Perform object detection
    results = model(image_pil)

    # Extract bounding boxes, labels, and confidence scores
    boxes = results.pred[0][:, :4].detach().cpu().numpy()
    scores = results.pred[0][:, 4].detach().cpu().numpy()
    labels = results.pred[0][:, 5].detach().cpu().numpy().astype(int)

    # Create a copy of the image for cropping
    cropped_image = image.copy()

    # Visualize the bounding boxes and confidence scores
    for box, score, label in zip(boxes, scores, labels):
        if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Crop the region of interest
            cropped_image = image[y1:y2, x1:x2]

    # Save the cropped image
    output_path = 'output/roi.jpg'  # Replace with the desired output path
    cv2.imwrite(output_path, cropped_image)

    # Display the image with bounding boxes
    # cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)

except cv2.error as e:
    print(f"OpenCV error: {e}")
