import json
from src.grader import grade_answer
from src.ocr import extract_text_from_image



def run_pipeline():
    # Load dataset
    with open("data/processed/valid_dataset.json", "r") as f:
        dataset = json.load(f)

    sample = dataset[0]

    print("\nQUESTION:")
    print(sample["question"])

    concepts = sample["concepts"]

    #OCR INPUT
    image_path = "data/Edtech_grading_file.jpeg"

    student_answer = extract_text_from_image(image_path)

    print("\nOCR EXTRACTED TEXT:")
    print(student_answer)

    result = grade_answer(concepts, student_answer)

    print("\nRESULT:")
    print("Keyword Score:", round(result["keyword_score"], 2))
    print("Similarity Score:", round(result["similarity_score"], 2))
    print("Final Score:", round(result["final_score"], 2))

    print("\nMatched Concepts:")
    for c in result["matched_concepts"]:
        print(":heavy_check_mark:", c)

    print("\nMissing Concepts:")
    for c in concepts:
        if c not in result["matched_concepts"]:
            print("✘", c)



if __name__ == "__main__":
    run_pipeline()