try:
    from src.phase1.keyword_baseline import concept_match_score
    from src.phase1.tfidf_advanced import concept_similarity_score
    from src.phase1.utils import final_score
except ImportError:
    from src.keyword_baseline import concept_match_score
    from src.tfidf_advanced import concept_similarity_score
    from src.utils import final_score

def grade_answer(concepts, student_answer):
    k_score, matched, all_concepts = concept_match_score(concepts, student_answer)
    sim_score = concept_similarity_score(concepts, student_answer)

    final = final_score(k_score, sim_score)

    return {
        "keyword_score": k_score,
        "similarity_score": sim_score,
        "final_score": final,
        "matched_concepts": matched,
        "total_concepts": all_concepts
    }