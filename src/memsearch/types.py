from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Evidence:
    id: str
    title: str
    sentence_id: int
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    sentences: tuple[str, ...]

    def evidence(self) -> list[Evidence]:
        return [Evidence(f"{self.id}:{i}", self.title, i, s)
                for i, s in enumerate(self.sentences) if s.strip()]


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class Labels:
    answer: str
    supporting_ids: tuple[str, ...]


@dataclass(frozen=True)
class MemoryItem:
    text: str
    source_ids: tuple[str, ...]
    memory_id: str = ""


@dataclass
class MemoryState:
    items: list[MemoryItem] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    next_id: int = 0

    def payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Action:
    kind: str
    query: str = ""
    answer: str = ""
    evidence_ids: tuple[str, ...] = ()


@dataclass
class Call:
    role: str
    messages: list[dict]
    response: str
    input_tokens: int
    output_tokens: int
    token_unit: str
    finish_reason: str | None = None
    decoding_mode: str = "text"


@dataclass
class Episode:
    task_id: str
    question: str
    answer: str
    evidence_ids: list[str]
    memory: MemoryState
    retrieved_ids: list[str]
    steps: list[dict]
    calls: list[Call]
    errors: list[str]
    stop_reason: str
    searches: int

    def payload(self) -> dict:
        return asdict(self)
