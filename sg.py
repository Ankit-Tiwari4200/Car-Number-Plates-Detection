import cv2
import pytesseract

# Path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change this path as per your installation

def read_number_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # Apply GaussianBlur to remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Use Otsu's binarization
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use pytesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(binary, lang='eng+Devanagari', config=custom_config)  # Add 'Devanagari' for Marathi

    # Print the recognized text
    print("Detected text:", text)

    return text

if __name__ == "__main__":
    # Filename of your image in the same directory
    image_filename = 'image 1 (1).jpeg'  # Replace with your actual image filename
    
    read_number_plate(image_filename)
