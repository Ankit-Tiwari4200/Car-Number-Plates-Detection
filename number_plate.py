import os
import cv2
import easyocr
import concurrent.futures
import re

# Path to the Haar Cascade file for plate detection
harcascade = "model/haarcascade_russian_plate_number.xml"

# Check if the Haar Cascade file exists
if not os.path.exists(harcascade):
    print("Error: Haar Cascade file not found.")
    exit()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the plate cascade classifier
plate_cascade = cv2.CascadeClassifier(harcascade)

# Initialize EasyOCR reader for Hindi and English
reader = easyocr.Reader(['en', 'hi'])

def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def is_valid_plate(w, h):
    aspect_ratio = w / h
    return 2 <= aspect_ratio <= 5 and w * h >= min_area

def is_valid_text(text):
    plate_pattern = re.compile(r'^[a-zA-Z0-9\s\W_\u0900-\u097F]+$')  # Allow alphanumeric, special symbols, and Devanagari script characters
    return bool(plate_pattern.match(text))

def process_frame(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 20))

    all_text = []  # Accumulate all text detected in the frame

    for (x, y, w, h) in plates:
        if is_valid_plate(w, h):
            img_roi = frame[y: y + h, x:x + w]
            roi_preprocessed = preprocess_image(img_roi)
            result = reader.readtext(roi_preprocessed, detail=0)  # Specify detail=0 for only text

            for res in result:
                dev_text = res
                if is_valid_text(dev_text):
                    all_text.append(dev_text)  # Append each valid text to the list

    # Join all the text into one line
    all_text_line = ' '.join(all_text)

    # Print or write to file the concatenated text
    if all_text_line:
        print("Concatenated Text:", all_text_line)
        with open('generated_text.txt', 'a', encoding='utf-8') as file:
            file.write(all_text_line + '\n')

    return frame

min_area = 1000

def reset_text_file():
    """Clears the content of the generated_text.txt file."""
    open('generated_text.txt', 'w').close()
    print("Text file has been reset.")

def process_frames(frames):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = executor.map(process_frame, frames)
        return list(processed_frames)

while True:
    ret, img = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Resize the frame for faster processing
    img = cv2.resize(img, (720, 580))

    processed_frames = process_frames([img])

    for frame in processed_frames:
        cv2.imshow("Results", frame)

    # Check for a specific key to reset the text file
    if cv2.waitKey(1) & 0xFF == ord('r'):
        reset_text_file()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
