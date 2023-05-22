import cv2

def detect_image(output_folder, file_name):
    # Callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal frame
        nonlocal capturing

        if event == cv2.EVENT_LBUTTONDOWN:  # Left button clicked
            # Save the frame as an image
            image_path = f"{output_folder}/{file_name}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved at: {image_path}")
            capturing = False

    capturing = True

    # Create a named window
    cv2.namedWindow("Camera")

    # Set mouse callback function
    cv2.setMouseCallback("Camera", mouse_callback)

    camera = cv2.VideoCapture(2)  # Open the default camera (index 0)

    # Check if camera opened successfully
    if not camera.isOpened():
        print("Failed to open camera")
        return

    while capturing:
        # Read frame from the camera
        ret, frame = camera.read()

        if ret:
            # Display the frame in the "Camera" window
            cv2.imshow("Camera", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            break

    # Release the camera and destroy the window
    camera.release()
    cv2.destroyAllWindows()

# Set the output folder and file name
output_folder = "out_folder"
file_name = "image"

# Call the function to save the image
detect_image(output_folder, file_name)
