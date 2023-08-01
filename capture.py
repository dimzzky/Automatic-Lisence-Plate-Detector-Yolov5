import cv2
import os

# Create the input folder if it doesn't exist
input_folder = 'input'
os.makedirs(input_folder, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Failed to open webcam")

# Initialize frame counter
frame_count = 0

while True:
    # Read frames from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check for 'C' or 'c' key press to capture and save the frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('C') or key == ord('c'):
        # Save the frame as an image
        image_path = os.path.join(input_folder, f"image.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved image: {image_path}")

        # Increment frame counter
        frame_count += 1

    # Check for 'q' key press to exit
    if key == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
