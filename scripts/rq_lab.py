from __future__ import annotations
"""
Tiny harness:
- Define a few gold questions with rough 'ticker hints' and keywords to match.
- Measures retrieval + calls generation & prints faithfulness.
This is just to demonstrate evaluation wiring; plug your own dataset.
"""
from src.trustgraph.indexes import retrieve
from src.trustgraph.generation import answer_with_evidence

GOLDS = [
    {
        "q": "What drove Apple's revenue change year-over-year in its latest 10-K? Cite the lines and key numbers.",
        "ticker": "AAPL",
        "keywords": ["revenue","year over year","segment"]
    },
    {
        "q": "Summarize NVIDIA gross margin trend and drivers, with table references.",
        "ticker": "NVDA",
        "keywords": ["gross margin","trend","drivers"]
    }
]

def main():
    for g in GOLDS:
        print("\nQ:", g["q"])
        docs = retrieve(g["q"], k=10)
        ids = [d["id"] for d in docs]
        print("Top docs:", ids[:5])
        qa = answer_with_evidence(g["q"], ticker_hint=g["ticker"])
        print("Faithfulness:", qa.faithfulness, "| Abstained:", qa.abstained)
        print("Answer:\n", qa.answer_markdown[:600], "...\n")

if __name__ == "__main__":
    main()
