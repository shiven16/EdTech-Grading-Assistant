import pytesseract
import cv2



def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding (important for OCR)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh



def extract_text_from_image(image_path):
    processed_img = preprocess_image(image_path)

    # pytesseract expects PIL image or array
    text = pytesseract.image_to_string(processed_img)

    return text