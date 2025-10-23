import os
import logging
import re

from pymilvus import utility, Collection
from typing import List, Dict, Any, Optional

# Logging instance for this module
_log = logging.getLogger(__name__)

def query_milvus(
    collection_name: str,
    embedding: List[float],
    output_fields: Optional[List[str]] = None,
    top_k: int = 5,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    _log.debug(f"Querying Milvus collection '{collection_name}'")    

    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist.")

    collection = Collection(collection_name)
    collection.load()

    if params is None:
        params = {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        }

    results = collection.search(
        data=[embedding],
        anns_field="vector",
        param=params,
        limit=top_k,
        expr=None,
        output_fields=output_fields,
        timeout=timeout,
    )

    hits = results[0]  # type: ignore
    return [
        dict(
            {
                "id": hit.id,
                "distance": hit.distance,
            },
            **({field: hit.entity.get(field) for field in output_fields} if output_fields else {})
        )
        for hit in hits
    ]

async def retrieve_context(vector: List[float], db_type: str, collection: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve context from the database based on the provided vector.
    Args:
        vector (List[float]): The vector to search for.
        db_type (str): The type of database to query (e.g., "milvus").
        top_k (int): The number of top results to return.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the retrieved context.
    """
    if db_type.lower() == "milvus":
        results = query_milvus(
            collection, 
            vector,
            ["content", "source", "headings", "page_number_min", "page_number_max"],
            top_k,
        )
        # return [doc["content"] for doc in results]
        return results
    else:
        raise ValueError(f"Unsupported db_type: {db_type}")
