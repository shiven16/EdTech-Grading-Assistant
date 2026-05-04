from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from src.phase1.preprocess import clean_text, remove_stopwords
except ImportError:
    from src.preprocess import clean_text, remove_stopwords

def preprocess(text):
    text = clean_text(text)
    words = remove_stopwords(text)
    return " ".join(words)


def split_into_chunks(text):
    # simple split by commas / sentences
    import re
    chunks = re.split(r'[.,]', text)
    return [c.strip() for c in chunks if c.strip()]


def concept_similarity_score(concepts, student_answer):
    student_chunks = split_into_chunks(student_answer)

    scores = []

    for concept in concepts:
        concept_clean = preprocess(concept)
        if not concept_clean.strip():
            continue

        best_score = 0

        for chunk in student_chunks:
            chunk_clean = preprocess(chunk)
            if not chunk_clean.strip():
                continue

            try:
                vectorizer = TfidfVectorizer(ngram_range=(1,2))
                vectors = vectorizer.fit_transform([concept_clean, chunk_clean])
                sim = cosine_similarity(vectors[0], vectors[1])[0][0]
                best_score = max(best_score, sim)
            except ValueError:
                pass

        scores.append(best_score)

    return sum(scores) / len(scores) if scores else 0