"""Optional verl custom reward hook for FINAL-ANSWER-only warm-up.

This does NOT implement multi-turn rollout, memory actions or their credit
assignment. Use the reference trainer until a pinned verl AgentLoop is adapted.
"""
from .backends import parse_json
from .metrics import answer_scores


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        result = parse_json(solution_str)
        answer = result["answer"]
        if not isinstance(answer, str):
            return 0.0
        return answer_scores(answer, ground_truth)[1]
    except (ValueError, KeyError, TypeError):
        return 0.0
