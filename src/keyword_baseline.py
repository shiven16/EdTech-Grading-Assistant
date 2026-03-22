from src.preprocess import clean_text

def concept_match_score(concepts, student_answer):
    student_words = set(clean_text(student_answer).split())

    matched = []

    for concept in concepts:
        concept_words = set(clean_text(concept).split())

        # measure overlap
        overlap = concept_words & student_words

        # threshold (important)
        if len(overlap) / len(concept_words) > 0.4:
            matched.append(concept)

    score = len(matched) / len(concepts) if concepts else 0

    return score, matched, concepts