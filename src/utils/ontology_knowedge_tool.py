import json
import logging

from .utils import get_weaviate_client, hybrid_search

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OntologyKnowledgeTool")


def extract_json_from_text(text):
    """```json
    {
    "extracted_terms": {
    "1": [
      {
        "entity": "mouse",
        "label": "ANIMAL_SPECIES",
        "sentence": "Here we report a comprehensive and high-resolution transcriptomic and spatial cell-type atlas for the whole adult mouse brain.",
        "start": 94,
        "end": 99,
        "paper_location": "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain",
        "paper_title": "Check for updates",
        "doi": null
      }
    ],
    "2": [
      {
        "entity": "isocortex",
        "label": "ANATOMICAL_REGION",
        "sentence": "Telencephalon consists of five major brain structures: isocortex, hippocampal formation (HPF), olfactory areas (OLF), cortical subplate (CTXsp) and cerebral nuclei (CNU).",
        "start": 54,
        "end": 63,
        "paper_location": "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain",
        "paper_title": "Check for updates",
        "doi": null
      },
      {
        "entity": "hippocampal formation",
        "label": "ANATOMICAL_REGION",
        "sentence": "Telencephalon consists of five major brain structures: isocortex, hippocampal formation (HPF), olfactory areas (OLF), cortical subplate (CTXsp) and cerebral nuclei (CNU).",
        "start": 65,
        "end": 85,
        "paper_location": "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain",
        "paper_title": "Check for updates",
        "doi": null
      },
      {
        "entity": "olfactory areas",
        "label": "ANATOMICAL_REGION",
        "sentence": "Telencephalon consists of five major brain structures: isocortex, hippocampal formation (HPF), olfactory areas (OLF), cortical subplate (CTXsp) and cerebral nuclei (CNU).",
        "start": 94,
        "end": 110,
        "paper_location": "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain",
        "paper_title": "Check for updates",
        "doi": null
      },
      {
        "entity": "cortical subplate",
        "label": "ANATOMICAL_REGION",
        "sentence": "Telencephalon consists of five major brain structures: isocortex, hippocampal formation (HPF), olfactory areas (OLF), cortical subplate (CTXsp) and cerebral nuclei (CNU).",
        "start": 113,
        "end": 130,
        "paper_location": "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain",
        "paper_title": "Check for updates",
        "doi": null
      },
      {
        "entity": "cerebral nuclei",
        "label": "ANATOMICAL_REGION",
        "sentence": "Telencephalon consists of five major brain structures: isocortex, hippocampal formation (HPF), olfactory areas (OLF), cortical subplate (CTXsp) and cerebral nuclei (CNU).",
        "start": 139,
        "end": 155,
        "paper_location": "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain",
        "paper_title": "Check for updates",
        "doi": null
      }
    ],
    "3": [
      {
        "entity": "astrocytes",
        "label": "CELL_TYPE",
        "sentence": "The Astro-Epen class is the most complex, containing ten subclasses, five of which represent astrocytes that are specific to different brain regions.",
        "start": 79,
        "end": 89,
        "paper_location": "Non-neuronal and immature neuronal cell types",
        "paper_title": "Check for updates",
        "doi": null
      },
      {
        "entity": "oligodendrocytes",
        "label": "CELL_TYPE",
        "sentence": "The OPC-Oligo class contains two subclasses, oligodendrocyte precursor cells (OPC) and oligodendrocytes.",
        "start": 90,
        "end": 106,
        "paper_location": "Non-neuronal and immature neuronal cell types",
        "paper_title": "Check for updates",
        "doi": null
      },
      {
        "entity": "microglia",
        "label": "CELL_TYPE",
        "sentence": "The Immune class consists of 5 subclasses: microglia, border-associated macrophages (BAM), monocytes, dendritic cells (DC) and lymphoid cells, which contains B cells, T cells, natural killer (NK) cells and innate lymphoid cells (ILC).",
        "start": 41,
        "end": 49,
        "paper_location": "Non-neuronal and immature neuronal cell types",
        "paper_title": "Check for updates",
        "doi": null
      }
    ]
    }
    }
    ``` 

    (Note: This output contains representative examples from the input text. The complete response will include further identified entities systematically extracted from the entire provided document.)
    """
    brace_stack = []
    json_start = None
    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                json_start = i
            brace_stack.append('{')
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:
                    json_end = i + 1
                    json_str = text[json_start:json_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Found potential JSON but failed to parse: {e}")
    raise ValueError("No valid JSON object found in the input.")


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
