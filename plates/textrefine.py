import re

# Define a regex pattern for Devanagari number plates
# This pattern looks for any line ending with one or more Devanagari numerals
devnagari_pattern = re.compile(r'.*[\u0966-\u096F]+$')

def is_valid_number_plate(text):
    """Check if the extracted text matches the Devanagari number plate pattern."""
    match = devnagari_pattern.match(text)
    if match:
        print(f"Match found: {text}")
    else:
        print(f"No match: {text}")
    return bool(match)

def extract_valid_number_plates(file_path):
    valid_plates = []
    detected_texts = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.strip()
            print(f"Processing line: {line}")  # Debugging output
            detected_texts.append(line)
            if is_valid_number_plate(line):
                valid_plates.append(line)
                print(f"Valid Number Plate: {line}")

    # Save all detected texts to a new file
    with open('detected_texts.txt', 'w', encoding='utf-8') as detected_file:
        for text in detected_texts:
            detected_file.write(text + '\n')

    # Save the valid number plates to a new file
    with open('valid_number_plates.txt', 'w', encoding='utf-8') as valid_file:
        for plate in valid_plates:
            valid_file.write(plate + '\n')

if __name__ == "__main__":
    # Specify the full path to the file
    file_path = 'generated_text.txt'
    extract_valid_number_plates(file_path)
