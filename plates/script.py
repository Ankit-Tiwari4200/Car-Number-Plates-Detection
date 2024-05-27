import cv2
import os
import pytesseract

# Path to the image containing Devanagari text
image_path = r'C:\Users\tiwar\OneDrive\Desktop\cv2 project\one more\Car-Number-Plates-Detection\plates\scaned_img_3.jpg'
pytesseract.pytesseract.tesseract_cmd = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform thresholding to obtain a binary image
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Use Tesseract OCR to extract text from the binary image
text = pytesseract.image_to_string(binary_image, lang='dev')

# Print the extracted text
print("Devanagari Text:", text)






####




import os
import cv2
import pytesseract
import concurrent.futures

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

# Devanagari to English numeral mapping
devanagari_to_english = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

def convert_devanagari_numbers(text):
    """Convert Devanagari numerals in the text to English numerals."""
    return ''.join(devanagari_to_english.get(char, char) for char in text)

def preprocess_for_ocr(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def process_frame(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            img_roi = frame[y: y + h, x:x + w]
            preprocessed = preprocess_for_ocr(img_roi)
            dev_text = pytesseract.image_to_string(preprocessed, lang='eng+hin', config='--psm 6')
            print("OCR Text:", dev_text)
            
            # Convert Devanagari numerals to English
            english_numerals = convert_devanagari_numbers(dev_text)
            print("Text with English Numerals:", english_numerals)
            
            cv2.putText(frame, english_numerals, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    return frame

min_area = 500

while True:
    ret, img = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    img = cv2.resize(img, (640, 480))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = executor.map(process_frame, [img])

        for frame in processed_frames:
            cv2.imshow("Results", frame)

    if cv2.waitKey(1000) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()




#
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

# Initialize EasyOCR reader for Devanagari
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
    devnagari_plate_pattern = re.compile(r'^[\u0900-\u097F0-9 ]+$')
    return bool(devnagari_plate_pattern.match(text))

def process_frame(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 20))

    for (x, y, w, h) in plates:
        if is_valid_plate(w, h):
            img_roi = frame[y: y + h, x:x + w]
            roi_preprocessed = preprocess_image(img_roi)
            result = reader.readtext(roi_preprocessed)

            for res in result:
                dev_text = res[1]
                if is_valid_text(dev_text):
                    print("Devanagari Text:", dev_text)
                    with open('generated_text.txt', 'a', encoding='utf-8') as file:
                        file.write(dev_text + '\n')
                    cv2.putText(frame, dev_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame

def reset_text_file():
    """Clears the content of the generated_text.txt file."""
    open('generated_text.txt', 'w').close()
    print("Text file has been reset.")

min_area = 500

while True:
    ret, img = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Resize the frame for faster processing
    img = cv2.resize(img, (640, 480))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = executor.map(process_frame, [img])

        for frame in processed_frames:
            cv2.imshow("Results", frame)

    # Check for a specific key to reset the text file
    if cv2.waitKey(1) & 0xFF == ord('r'):
        reset_text_file()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()


#
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

# Initialize EasyOCR reader for Marathi
reader = easyocr.Reader(['mr'])

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
    marathi_plate_pattern = re.compile(r'^[\u0900-\u097F0-9 ]+$')
    return bool(marathi_plate_pattern.match(text))

def process_frame(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 20))

    for (x, y, w, h) in plates:
        if is_valid_plate(w, h):
            img_roi = frame[y: y + h, x:x + w]
            roi_preprocessed = preprocess_image(img_roi)
            result = reader.readtext(roi_preprocessed, detail=0)  # Specify detail=0 for only text

            for res in result:
                marathi_text = res
                if is_valid_text(marathi_text):
                    print("Marathi Text:", marathi_text)
                    with open('generated_text.txt', 'a', encoding='utf-8') as file:
                        file.write(marathi_text + '\n')
                    cv2.putText(frame, marathi_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame

def reset_text_file():
    """Clears the content of the generated_text.txt file."""
    open('generated_text.txt', 'w').close()
    print("Text file has been reset.")

min_area = 500

while True:
    ret, img = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Resize the frame for faster processing
    img = cv2.resize(img, (640, 480))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = executor.map(process_frame, [img])

        for frame in processed_frames:
            cv2.imshow("Results", frame)

    # Check for a specific key to reset the text file
    if cv2.waitKey(1) & 0xFF == ord('r'):
        reset_text_file()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()


#
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
    devnagari_plate_pattern = re.compile(r'^[\u0900-\u097F0-9 ]+$')
    return bool(devnagari_plate_pattern.match(text))

def process_frame(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 20))

    for (x, y, w, h) in plates:
        if is_valid_plate(w, h):
            img_roi = frame[y: y + h, x:x + w]
            roi_preprocessed = preprocess_image(img_roi)
            result = reader.readtext(roi_preprocessed, detail=0)  # Specify detail=0 for only text

            for res in result:
                dev_text = res
                if is_valid_text(dev_text):
                    print("Devanagari Text:", dev_text)
                    with open('generated_text.txt', 'a', encoding='utf-8') as file:
                        file.write(dev_text + '\n')
                    cv2.putText(frame, dev_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame

def reset_text_file():
    """Clears the content of the generated_text.txt file."""
    open('generated_text.txt', 'w').close()
    print("Text file has been reset.")

min_area = 500

while True:
    ret, img = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Resize the frame for faster processing
    img = cv2.resize(img, (640, 480))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = executor.map(process_frame, [img])

        for frame in processed_frames:
            cv2.imshow("Results", frame)

    # Check for a specific key to reset the text file
    if cv2.waitKey(1) & 0xFF == ord('r'):
        reset_text_file()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()