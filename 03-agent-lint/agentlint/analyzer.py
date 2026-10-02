"""
Detect common multi-agent conversation failure patterns:

- loops: two agents repeating semantically similar messages back and forth
- drift: the conversation wandering away from the original task over time
- premature termination: the conversation ending without a resolved answer

Uses TF-IDF + cosine similarity for the offline default (see contextprune's
embeddings.py for the same pattern) -- swap in real sentence embeddings for
production use where nuanced paraphrase-level similarity matters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Message:
    index: int
    sender: str
    content: str


@dataclass
class Finding:
    kind: str            # "loop" | "drift" | "premature_termination"
    severity: str         # "warning" | "error"
    message_indices: list[int]
    description: str


UNRESOLVED_MARKERS = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "unable to determine", "cannot determine", "let's stop here",
    "i give up", "no further progress",
]


class ConversationAnalyzer:
    def __init__(self, loop_similarity_threshold: float = 0.85,
                 drift_drop_threshold: float = 0.4,
                 window: int = 4):
        self.loop_similarity_threshold = loop_similarity_threshold
        self.drift_drop_threshold = drift_drop_threshold
        self.window = window  # how many messages back to check for loops

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english")
        return vectorizer.fit_transform(texts).toarray()

    def analyze(self, task: str, messages: Sequence[Message]) -> list[Finding]:
        findings: list[Finding] = []
        if not messages:
            return findings

        all_texts = [task] + [m.content for m in messages]
        vectors = self._embed_all(all_texts)
        task_vec, msg_vecs = vectors[0:1], vectors[1:]

        findings.extend(self._detect_loops(messages, msg_vecs))
        findings.extend(self._detect_drift(task, messages, task_vec, msg_vecs))
        findings.extend(self._detect_premature_termination(messages))
        return findings

    def _detect_loops(self, messages: Sequence[Message], msg_vecs: np.ndarray) -> list[Finding]:
        findings = []
        seen_pairs = set()
        for i in range(len(messages)):
            for j in range(max(0, i - self.window), i):
                if (j, i) in seen_pairs:
                    continue
                sim = float(cosine_similarity(msg_vecs[i:i+1], msg_vecs[j:j+1])[0][0])
                if sim >= self.loop_similarity_threshold:
                    seen_pairs.add((j, i))
                    findings.append(Finding(
                        kind="loop",
                        severity="error",
                        message_indices=[messages[j].index, messages[i].index],
                        description=(
                            f"Message {messages[j].index} ({messages[j].sender}) and message "
                            f"{messages[i].index} ({messages[i].sender}) are {sim:.0%} similar -- "
                            f"possible repetition loop."
                        ),
                    ))
        return findings

    def _detect_drift(self, task: str, messages: Sequence[Message],
                       task_vec: np.ndarray, msg_vecs: np.ndarray) -> list[Finding]:
        findings = []
        sims = cosine_similarity(task_vec, msg_vecs)[0]
        if len(sims) < 3:
            return findings
        # Compare the average similarity of the first third vs. the last third.
        third = max(1, len(sims) // 3)
        early_avg = float(np.mean(sims[:third]))
        late_avg = float(np.mean(sims[-third:]))
        if early_avg > 0 and (early_avg - late_avg) / early_avg >= self.drift_drop_threshold:
            findings.append(Finding(
                kind="drift",
                severity="warning",
                message_indices=[messages[-third].index, messages[-1].index],
                description=(
                    f"Relevance to the original task dropped {((early_avg - late_avg) / early_avg):.0%} "
                    f"from the start of the conversation to the end (early={early_avg:.2f}, late={late_avg:.2f}) "
                    f"-- the agents may have drifted off-topic."
                ),
            ))
        return findings

    def _detect_premature_termination(self, messages: Sequence[Message]) -> list[Finding]:
        last = messages[-1]
        lowered = last.content.lower()
        if any(marker in lowered for marker in UNRESOLVED_MARKERS):
            return [Finding(
                kind="premature_termination",
                severity="warning",
                message_indices=[last.index],
                description=(
                    f"Conversation ended on message {last.index} ({last.sender}) with an "
                    f"unresolved-sounding statement: \"{last.content[:80]}\""
                ),
            )]
        return []
