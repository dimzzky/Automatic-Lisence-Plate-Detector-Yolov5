import easyocr

# Set the language(s) you want to use for OCR
language = 'en'  # Replace with the language code(s) you need, e.g., 'en' for English

# Create an EasyOCR reader object
reader = easyocr.Reader([language], gpu=False)  # Set gpu=False to use CPU

# Path to the image you want to perform OCR on
image_path = 'test_images/images (1).jpeg'  # Replace with the path to your image

# Perform OCR on the image
results = reader.readtext(image_path)

# Process the OCR results
for (bbox, text, score) in results:
    # Extract the coordinates
    x1, y1, x2, y2 = bbox

    # Print the text and confidence score
    print(f'Text: {text}, Score: {score}')

    # You can also perform further processing on the text or bounding box coordinates as needed
