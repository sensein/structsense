import json
import logging

from .utils import get_weaviate_client, hybrid_search, extract_json_from_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OntologyKnowledgeTool")




def OntologyKnowledgeTool(data, search_key=["entity", "label"]):
    """Extracts ontology metadata for any structured data, performs hybrid search,
    and returns structured ontology knowledge grouped by a configurable search key.

    Args:
        data (Dict): Any structured dictionary containing lists of records.
        search_key (str): The key to look up in each item for search (e.g., 'entity').

    Returns:
        str: A plain, long string containing ontology knowledge.
    """
    response_text = ""

    if not data:
        logger.error("No input data provided.")
        return "Error: Missing required input: data."

    # Dynamically locate nested dictionaries that contain lists of dicts
    data = extract_json_from_text(data)

    term_collections = []
    if isinstance(data, dict):
        for _, val in data.items():
            if isinstance(val, dict):
                for inner_val in val.values():
                    if isinstance(inner_val, list):
                        term_collections.append(inner_val)

    if not term_collections:
        logger.error("Unexpected format: no list of terms found in input data.")
        return "Error: Invalid input structure."

    for term_group in term_collections:
        for term in term_group:
            if not isinstance(term, dict):
                logger.warning(f"Skipping non-dictionary term: {term}")
                continue

            # Combine multiple search keys into one string
            combined_query_parts = [str(term.get(k)).strip() for k in search_key if term.get(k)]
            query = " ".join(combined_query_parts)

            if not query:
                logger.warning(f"Skipping term due to missing '{search_key}': {term}")
                continue

            response_text += f"Search Term: {query}"

            try:
                # Perform hybrid search for the query term
                client = get_weaviate_client()
                search_results = hybrid_search(client, query)
                client.close()

                # Append search results to response text
                if search_results:
                    response_text += "Knowledge: "
                    for res in search_results:
                        response_text += (
                            f"[Label: {res.get('label', 'N/A')}, "
                            f"Definition: {res.get('definition', 'No definition available')}, "
                            f"Ontology: {res.get('ontology', 'N/A')}, "
                            f"Class URI: {res.get('class_uri', 'N/A')}]. "
                        )

            except Exception as e:
                logger.error(f"Hybrid search failed for query '{query}': {e}")
                response_text += "Search Results: Failed to retrieve search results. "

    return response_text.strip() if response_text else "No ontology knowledge found."
