from __future__ import annotations

import re

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from utils.embeddings import DEFAULT_MODE


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful PDF question answering assistant.\n"
        "Use only the provided context to answer the question.\n"
        "If the answer is not present in the context, say that you do not know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

WORD_PATTERN = re.compile(r"[a-zA-Z0-9]{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class LocalQAChain:
    """Simple local QA pipeline that answers from retrieved PDF sentences."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever

    def invoke(self, inputs: dict) -> dict:
        query = inputs.get("query", "")
        question = inputs.get("question", query)
        source_documents = self.retriever.invoke(query)
        answer = generate_local_answer(question, source_documents)

        return {
            "result": answer,
            "source_documents": source_documents,
        }


def build_qa_chain(retriever, mode: str = DEFAULT_MODE):
    """Build either the OpenAI QA chain or the local QA chain."""
    if mode == "openai":
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT},
        )

    return LocalQAChain(retriever)


def build_contextual_question(question: str, chat_history: list[dict], history_turns: int = 3) -> str:
    """Add recent user questions so simple follow-ups work better."""
    if not chat_history:
        return question

    recent_turns = chat_history[-history_turns:]
    conversation_lines = []

    for item in recent_turns:
        conversation_lines.append(f"Previous user question: {item['question']}")

    conversation_text = "\n".join(conversation_lines)
    return (
        "Conversation history:\n"
        f"{conversation_text}\n\n"
        f"Current user question:\n{question}"
    )


def tokenize_for_search(text: str) -> list[str]:
    """Create a lightweight token list for local scoring."""
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def split_into_sentences(text: str) -> list[str]:
    """Split chunk text into display-friendly sentences."""
    return [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]


def score_sentence(question: str, sentence: str, doc_rank: int) -> float:
    """Score how relevant a sentence is to the user question."""
    question_tokens = {token for token in tokenize_for_search(question) if token not in STOPWORDS}
    sentence_tokens = set(tokenize_for_search(sentence))

    if not question_tokens or not sentence_tokens:
        return 0.0

    overlap = question_tokens.intersection(sentence_tokens)
    number_overlap = {
        token for token in question_tokens
        if any(character.isdigit() for character in token) and token in sentence_tokens
    }

    score = len(overlap) * 3.0
    score += len(number_overlap) * 4.0
    score += max(0, 3 - doc_rank) * 0.8

    if question.lower() in sentence.lower():
        score += 6.0

    if len(sentence) < 25:
        score -= 0.5

    return score


def generate_local_answer(question: str, source_documents: list) -> str:
    """Build a simple extractive answer from retrieved sentences."""
    if not source_documents:
        return "I could not find any relevant content in the indexed PDFs."

    scored_sentences: list[tuple[float, str]] = []

    for doc_rank, document in enumerate(source_documents):
        sentences = split_into_sentences(document.page_content)
        for sentence in sentences[:8]:
            sentence_score = score_sentence(question, sentence, doc_rank)
            if sentence_score > 0:
                scored_sentences.append((sentence_score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    if not scored_sentences:
        best_chunk = source_documents[0].page_content.strip().replace("\n", " ")
        if len(best_chunk) > 280:
            best_chunk = best_chunk[:280].rstrip() + "..."
        return (
            "I found related content, but I could not extract a confident short answer. "
            f"The most relevant chunk says: {best_chunk}"
        )

    top_score = scored_sentences[0][0]
    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()

    for score, sentence in scored_sentences:
        normalized_sentence = sentence.lower()
        if normalized_sentence in seen_sentences:
            continue

        if selected_sentences and score < top_score * 0.65:
            continue

        seen_sentences.add(normalized_sentence)
        selected_sentences.append(sentence)

        if len(selected_sentences) == 2:
            break

    if selected_sentences:
        return " ".join(selected_sentences)

    best_chunk = source_documents[0].page_content.strip().replace("\n", " ")
    if len(best_chunk) > 280:
        best_chunk = best_chunk[:280].rstrip() + "..."

    return (
        "I found related content, but I could not extract a confident short answer. "
        f"The most relevant chunk says: {best_chunk}"
    )


def format_source_documents(source_documents: list) -> list[dict]:
    """Prepare source chunks for display in the Streamlit UI."""
    formatted_sources: list[dict] = []
    seen_citations: set[tuple[str, int, int]] = set()

    for document in source_documents:
        source_name = document.metadata.get("source", "Unknown source")
        page = int(document.metadata.get("page", 0))
        chunk_number = int(document.metadata.get("chunk_number", 0))
        citation_key = (source_name, page, chunk_number)

        if citation_key in seen_citations:
            continue

        seen_citations.add(citation_key)
        preview = document.page_content.strip().replace("\n", " ")

        if len(preview) > 350:
            preview = preview[:350].rstrip() + "..."

        formatted_sources.append(
            {
                "citation": f"{source_name} | Page {page} | Chunk {chunk_number}",
                "content": preview,
            }
        )

    return formatted_sources


def ask_pdf_question(qa_chain, question: str, chat_history: list[dict]) -> dict:
    """Run a question against the QA chain and return answer + citations."""
    if qa_chain is None:
        raise ValueError("The question answering chain is not ready yet.")

    contextual_question = build_contextual_question(question, chat_history)
    result = qa_chain.invoke(
        {
            "query": contextual_question,
            "question": question,
        }
    )

    return {
        "answer": result.get("result", "No answer was generated."),
        "sources": format_source_documents(result.get("source_documents", [])),
    }
