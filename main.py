import json
from src.grader import grade_answer


def run_pipeline():
    # Load ONLY valid dataset
    with open("data/processed/valid_dataset.json", "r") as f:
        dataset = json.load(f)

    # Pick first valid dataset
    sample = dataset[0]

    print("\nQUESTION:")
    print(sample["question"])

    concepts = sample["concepts"]

    # Sample student answer
    student_answer = (
        "To replicate the experiment, you need to know how much vinegar was used, "
        "what type of vinegar was used, what materials were tested, "
        "the surface area of the materials, and how long each sample was rinsed."
    )

    result = grade_answer(concepts, student_answer)

    print("\nSTUDENT ANSWER:")
    print(student_answer)

    print("\nRESULT:")
    print("Keyword Score:", round(result["keyword_score"], 2))
    print("Similarity Score:", round(result["similarity_score"], 2))
    print("Final Score:", round(result["final_score"], 2))

    print("\nMatched Concepts:")
    for c in result["matched_concepts"]:
        print("✔", c)

    print("\nMissing Concepts:")
    for c in concepts:
        if c not in result["matched_concepts"]:
            print("✘", c)


if __name__ == "__main__":
    run_pipeline()