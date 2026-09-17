from __future__ import annotations
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import weaviate
from weaviate.classes.query import MetadataQuery
from langchain_openai import ChatOpenAI  # 
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# OPENAI_API_KEY is loaded from the environment; never hard-code secrets here.

def _build_interaction_query(drug1: str, drug2: str) -> str:
    return (
        f"{drug1} {drug2} interaction effects adverse events contraindication "
        f"bleeding toxicity pharmacokinetics mechanism risk"
    )

def retrieve_related_entries(
    drug1: str,
    drug2: str,
    *,
    collection_name: str = "TextChunk",
    text_property: str = "content",
    limit: int = 5,
    weaviate_version: str = "latest",
    persistence_data_path: str = "./weaviate_data",
) -> List[Dict[str, Any]]:
    query = _build_interaction_query(drug1, drug2)
    client = weaviate.connect_to_embedded(
        version=weaviate_version,
        persistence_data_path=persistence_data_path,
        environment_variables={"LOG_LEVEL": "error"},
    )

    try:
        col = client.collections.use(collection_name)
        try:
            resp = col.query.hybrid(
                query=query,
                limit=limit,
            )
            objects = resp.objects
            results = []
            for o in objects:
                props = o.properties or {}
                results.append(
                    {
                        "text": props.get(text_property, "") or "",
                        "score": getattr(getattr(o, "metadata", None), "score", None),
                        "props": props,
                    }
                )
            # If hybrid returns usable text, keep it.
            if any(r["text"].strip() for r in results):
                return results
        except Exception:
            pass

        # 2) Fallback: BM25 keyword search (works without vectors)
        resp = col.query.bm25(
            query=query,
            query_properties=[text_property],
            return_metadata=MetadataQuery(score=True),
            limit=limit,
        )  # 

        results = []
        for o in resp.objects:
            props = o.properties or {}
            results.append(
                {
                    "text": props.get(text_property, "") or "",
                    "score": o.metadata.score if o.metadata else None,
                    "props": props,
                }
            )
        return results

    finally:
        client.close()


def summarize_interaction_effects(
    drug1: str,
    drug2: str,
    *,
    collection_name: str = "TextChunk",
    text_property: str = "content",
    limit: int = 12,
    model: str = "gpt-5-nano",
    persistence_data_path: str = "./weaviate_data",
) -> str:
    hits = retrieve_related_entries(
        drug1,
        drug2,
        collection_name=collection_name,
        text_property=text_property,
        limit=limit,
        persistence_data_path=persistence_data_path,
    )
    chunks = [h["text"].strip() for h in hits if h["text"].strip()]
    if not chunks:
        return "No related entries found in Weaviate for that drug pair."

    context = "\n\n---\n\n".join(chunks[:limit])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize biomedical information about drug–drug interactions.\n"
                "Use both the provided context and your own validated pharmacological knowledge.\n"
                "If the context conflicts with established knowledge, note the discrepancy.\n"
                "Do not give medical advice or treatment guidance. Focus on mechanisms, risks, and interaction relevance.\n"
                "Output must be a single paragraph written in clear technical language."
                "The paragraph must begin as a whole sentence, written as if the reader doesn't know what the paragraph is about.",
            ),
            (
                "user",
                "Drug pair: {drug1} + {drug2}\n\n"
                "Task: Write a single paragraph summarizing the interaction between these two drugs. "
                "Use the provided context and your broader biomedical knowledge. "
                "Explain the type of interaction, likely mechanisms, pharmacokinetic or pharmacodynamic factors, "
                "and any known clinical risks or adverse outcomes. "
                "If the context mentions specific findings, incorporate them. "
                "If some interaction details are not discussed in the context, rely on your own knowledge.\n\n"
                "Context:\n{context}"
            ),
        ]
    )

    llm = ChatOpenAI(model=model)  # 
    msg = llm.invoke(prompt.format_messages(drug1=drug1, drug2=drug2, context=context))
    return msg.content


if __name__ == "__main__":
    print(summarize_interaction_effects("ibuprofen", "warfarin"))
