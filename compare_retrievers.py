"""Compare the six retrievers on the scraped pages. Run: python compare_retrievers.py

A question counts as a hit when the retrieved context contains the expected passage,
so the score measures retrieval only, not the quality of the LLM's answer.
Scrape the three Wikipedia pages below first (python main.py --url ...).
"""

import json
import re
import time

from llama_index.core.utils import get_tokenizer

from modules.data_processing import load_documents
from modules.llm_interface import create_watsonx_embedding, create_watsonx_llm
from modules.retrievers import KnowledgeBase, RETRIEVER_NAMES
from modules.router import route_query

EIFFEL = "https://en.wikipedia.org/wiki/Eiffel_Tower"
PHOTO = "https://fr.wikipedia.org/wiki/Photosynth%C3%A8se"
BEE = "https://en.wikipedia.org/wiki/Honey_bee"

# (question, expected page, regex the context must contain; None = only the page is checked)
QUESTIONS = [
    ("How tall is the Eiffel Tower?", EIFFEL, r"330 metres"),
    ("How much paint is applied to the Eiffel Tower, and how often?", EIFFEL, r"60 tons of paint"),
    ("Quel architecte a contribué à la conception de la tour Eiffel ?", EIFFEL, r"asked Stephen Sauvestre"),
    ("Chez les plantes, dans quelle membrane se trouvent les protéines contenant la chlorophylle ?",
     PHOTO, r"membrane des thylakoïdes"),
    ("What happens to carbon dioxide in the Calvin cycle?",
     PHOTO, r"dioxyde de carbone atmosphérique est fixé"),
    ("Rubisco algues marines vitesse maximum", PHOTO, r"quart de leur vitesse maximum"),
    ("Comment les abeilles se communiquent-elles l'emplacement de la nourriture ?", BEE, r"waggle dance"),
    ("What is the biggest threat to honey bees in the United States?",
     BEE, r"Varroa mites are arguably the biggest threat"),
    ("Give me an overview of the article about honey bees.", BEE, None),
]


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main():
    documents = load_documents()
    missing = {EIFFEL, PHOTO, BEE} - {d.doc_id for d in documents}
    assert not missing, f"Scrape these pages first: {missing}"
    texts = {d.doc_id: squash(d.text) for d in documents}
    for question, page, pattern in QUESTIONS:  # a wrong expectation would fake a miss
        assert pattern is None or re.search(pattern, texts[page], re.I), (question, pattern)

    tokenize = get_tokenizer()
    llm = create_watsonx_llm()
    # separate Chroma folder so the running app's collections are not replaced
    kb = KnowledgeBase(documents, create_watsonx_embedding(), llm, "chroma_db_eval")
    print(f"RESULT {len(documents)} pages, {len(kb.chunks)} chunks of 512 tokens", flush=True)

    results = {"build_seconds": {}, "questions": []}
    for name in RETRIEVER_NAMES:
        start = time.perf_counter()
        kb.retriever(name)
        results["build_seconds"][name] = round(time.perf_counter() - start, 1)
        print(f"RESULT built {name} in {results['build_seconds'][name]}s", flush=True)

    for question, page, pattern in QUESTIONS:
        row = {"question": question, "routed_to": route_query(question, llm), "retrievers": {}}
        for name in RETRIEVER_NAMES:
            start = time.perf_counter()
            nodes = kb.retriever(name).retrieve(question)
            seconds = time.perf_counter() - start
            context = squash(" ".join(n.node.get_content() for n in nodes))
            urls = [n.node.metadata.get("url") or n.node.ref_doc_id for n in nodes]
            hit = bool(re.search(pattern, context, re.I)) if pattern else page in urls
            row["retrievers"][name] = {
                "hit": hit,
                "right_page_share": round(urls.count(page) / len(urls), 2) if urls else 0,
                "nodes": len(nodes),
                "context_tokens": len(tokenize(context)),
                "seconds": round(seconds, 2),
            }
        results["questions"].append(row)
        marks = " ".join(f"{n}={'OK' if r['hit'] else '--'}" for n, r in row["retrievers"].items())
        print(f"RESULT {marks} | router={row['routed_to']} | {question}", flush=True)

    print("RESULT | retriever | hits | right page | nodes | context tokens | query s | build s |")
    count = len(QUESTIONS)
    for name in RETRIEVER_NAMES:
        rows = [q["retrievers"][name] for q in results["questions"]]
        means = {k: sum(r[k] for r in rows) / count
                 for k in ("right_page_share", "nodes", "context_tokens", "seconds")}
        print(f"RESULT | {name} | {sum(r['hit'] for r in rows)}/{count} "
              f"| {means['right_page_share']:.0%} | {means['nodes']:.1f} "
              f"| {means['context_tokens']:.0f} | {means['seconds']:.2f} "
              f"| {results['build_seconds'][name]} |")
    routed = sum(q["retrievers"][q["routed_to"]]["hit"] for q in results["questions"])
    print(f"RESULT | router (auto) | {routed}/{count} |")

    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
