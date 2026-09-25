import math
import re
from collections import Counter


def words(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class BM25:
    def __init__(self, documents):
        self.documents = documents
        self.tf = {k: Counter(words(d.title + " " + " ".join(d.sentences))) for k, d in documents.items()}
        self.lengths = {k: sum(v.values()) for k, v in self.tf.items()}
        self.avg_length = sum(self.lengths.values()) / max(len(self.tf), 1)
        self.df = Counter(t for terms in self.tf.values() for t in terms)

    def search(self, query, k=2, candidate_ids=None):
        candidates = self.documents.keys() if candidate_ids is None else candidate_ids
        scores = []
        for doc_id in candidates:
            score = 0.0
            for token in set(words(query)):
                freq = self.tf[doc_id][token]
                idf = math.log(1 + (len(self.tf) - self.df[token] + 0.5) / (self.df[token] + 0.5))
                denominator = freq + 1.2 * (0.25 + 0.75 * self.lengths[doc_id] / max(self.avg_length, 1))
                score += idf * freq * 2.2 / denominator
            scores.append((score, doc_id))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [self.documents[key] for score, key in scores[:k] if score > 0]
