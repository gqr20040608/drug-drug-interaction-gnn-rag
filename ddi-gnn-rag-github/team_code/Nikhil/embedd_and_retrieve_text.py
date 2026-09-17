from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import weaviate
from weaviate.classes.config import Property, DataType
from langchain_text_splitters import RecursiveCharacterTextSplitter


def store_text_embedded_weaviate(
    text: str,
    *,
    collection_name: str = "TextChunk",
    source_id: str = "doc-1",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    weaviate_version: str = "latest",
    persistence_data_path: str = "./weaviate_data",
    extra_properties: Optional[Dict[str, Any]] = None,
) -> int:
    Path(persistence_data_path).mkdir(parents=True, exist_ok=True)
    client = weaviate.connect_to_embedded(
        version=weaviate_version,
        persistence_data_path=persistence_data_path,
        environment_variables={
            "LOG_LEVEL": "error",
            "CLUSTER_GOSSIP_BIND_ADDRESS": "127.0.0.1",
            "CLUSTER_DATA_BIND_ADDRESS": "127.0.0.1",
            "CLUSTER_GOSSIP_ADVERTISE_ADDRESS": "127.0.0.1",
            "CLUSTER_DATA_ADVERTISE_ADDRESS": "127.0.0.1",
        },
    )

    try:
        if not client.collections.exists(collection_name):
            client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source_id", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks: List[str] = splitter.split_text(text)

        extras = extra_properties or {}

        with client.batch.fixed_size(batch_size=128) as batch:
            for i, chunk in enumerate(chunks):
                props = {
                    "content": chunk,
                    "source_id": source_id,
                    "chunk_index": i,
                    **extras,
                }
                batch.add_object(
                    collection=collection_name,
                    properties=props,
                )

        return len(chunks)

    finally:
        client.close()


# with open("output.txt", "r") as f:
#     text = f.read()
#     print(len(text.split("\n")))
#     store_text_embedded_weaviate(text)
