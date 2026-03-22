import easyocr

# Initialize once (important for performance)
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    result = reader.readtext(image_path, detail=0)
    text = " ".join(result)
    return text