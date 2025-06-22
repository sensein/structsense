import json
import os
from pprint import pprint
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://data.bioontology.org"
API_KEY = os.environ.get("BIO_PORTAL")

def annotate_text(text: str, ontologies: list = None, semantic_types: list = None,
                  expand_semantic_types_hierarchy: bool = False,
                  expand_class_hierarchy: bool = False, class_hierarchy_max_level: int = 0,
                  expand_mappings: bool = False, stop_words: list = None,
                  minimum_match_length: int = None, exclude_numbers: bool = False,
                  whole_word_only: bool = True, exclude_synonyms: bool = False,
                  longest_only: bool = False, auth_method: str = "query"):
    """
    Examines text input and returns relevant classes from the specified Annotator API endpoint.
    Supports getting the API key from a .env file or directly as an argument.
    Authentication via query parameter or Authorization header is supported.

    Args:
        text: The input text to be annotated.
        ontologies: A list of ontology IDs to filter the annotation results.
        semantic_types: A list of semantic types to filter the annotation results.
        expand_semantic_types_hierarchy: Whether to include child semantic types (default: False).
        expand_class_hierarchy: Whether to include ancestor classes (default: False).
        class_hierarchy_max_level: The maximum depth of the class hierarchy to consider (default: 0).
        expand_mappings: Whether to include manual mappings (UMLS, REST, CUI, OBOXREF) (default: False).
        stop_words: A list of words to exclude from annotation (case-insensitive).
        minimum_match_length: The minimum length of a matched term to be included in the results.
        exclude_numbers: Whether to exclude matches that are purely numbers (default: False).
        whole_word_only: Whether to only match whole words (default: True).
        exclude_synonyms: Whether to exclude synonymous matches (default: False).
        longest_only: Whether to return only the longest match for a given phrase (default: False).
        auth_method: The method for providing the API key ("query" or "header") (default: "query").


    Returns:
        A dictionary or JSON object containing the annotation results.

    Raises:
        requests.exceptions.RequestException: If there is an error during the API request.
        ValueError: If the 'auth_method' is not 'query' or 'header'.
        EnvironmentError: If the API key is not provided as an argument and not found in the .env file.
    """
    url = f"{BASE_URL}/annotator"
    params = {"text": text}
    headers = {}

    if auth_method == "query":
        params["apikey"] = API_KEY
    elif auth_method == "header":
        headers["Authorization"] = f"apikey token={API_KEY}"
    else:
        raise ValueError("Invalid auth_method. Must be 'query' or 'header'.")

    if ontologies:
        params["ontologies"] = ",".join(ontologies)
    if semantic_types:
        params["semantic_types"] = ",".join(semantic_types)
    if expand_semantic_types_hierarchy is True:
        params["expand_semantic_types_hierarchy"] = "true"
    if expand_class_hierarchy is True:
        params["expand_class_hierarchy"] = "true"
    if class_hierarchy_max_level is not None:
        params["class_hierarchy_max_level"] = str(class_hierarchy_max_level)
    if expand_mappings is True:
        params["expand_mappings"] = "true"
    if stop_words:
        params["stop_words"] = ",".join(stop_words)
    if minimum_match_length is not None:
        params["minimum_match_length"] = str(minimum_match_length)
    if exclude_numbers is True:
        params["exclude_numbers"] = "true"
    if whole_word_only is False:
        params["whole_word_only"] = "false"
    if exclude_synonyms is True:
        params["exclude_synonyms"] = "true"
    if longest_only is True:
        params["longest_only"] = "true"

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except EnvironmentError as e:
        print(f"Error: {e}")
        return None
    
def recommend_ontologies(text: str, input_type: int = 1, output_type: int = 1,
                         max_elements_set: int = 3, wc: float = 0.55, wa: float = 0.15,
                         wd: float = 0.15, ws: float = 0.15, ontologies: list = None,
                         auth_method: str = "query", api_key: str = None):
    """
    Recommends appropriate ontologies for a given text or list of keywords
    using the BioPortal Recommender API endpoint.

    Args:
        input_text: The input text or comma-separated keywords.
        input_type: The type of input (1 for text, 2 for keywords) (default: 1).
        output_type: The type of output (1 for individual ontologies, 2 for ontology sets) (default: 1).
        max_elements_set: Maximum number of ontologies per set (only for output_type = 2) (default: 3).
        wc: Weight assigned to the ontology coverage criterion (default: 0.55).
        wa: Weight assigned to the ontology acceptance criterion (default: 0.15).
        wd: Weight assigned to the ontology detail criterion (default: 0.15).
        ws: Weight assigned to the ontology specialization criterion (default: 0.15).
        ontologies: A list of ontology IDs to limit the evaluation (default: None, all ontologies evaluated).
        auth_method: The method for providing the API key ("query" or "header") (default: "query").
        api_key: The API key. If None, it will be loaded from the .env file.

    Returns:
        A dictionary or JSON object containing the ontology recommendations.

    Raises:
        requests.exceptions.RequestException: If there is an error during the API request.
        ValueError: If 'input_type' or 'output_type' is not 1 or 2, or 'auth_method' is invalid,
                    or if weight values are not in the range [0, 1].
        EnvironmentError: If the API key is not provided and not found in the .env file.
    """
    url = f"{BASE_URL}/recommender"
    params = {"input": text}
    headers = {}

    if auth_method == "query":
        params["apikey"] = API_KEY
    elif auth_method == "header":
        headers["Authorization"] = f"apikey token={API_KEY}"
    else:
        raise ValueError("Invalid auth_method. Must be 'query' or 'header'.")


    # # Validate input and output types
    # if input_type not in [1, 2]:
    #     raise ValueError("Invalid input_type. Must be 1 (text) or 2 (keywords).")
    # if output_type not in [1, 2]:
    #     raise ValueError("Invalid output_type. Must be 1 (individual) or 2 (sets).")
    # if not all(0 <= weight <= 1 for weight in [wc, wa, wd, ws]):
    #     raise ValueError("Weight values (wc, wa, wd, ws) must be in the range [0, 1].")
    # if max_elements_set not in [2, 3, 4] and output_type == 2:
    #     raise ValueError("Invalid max_elements_set. Must be 2, 3, or 4 when output_type is 2.")


    # if ontologies:
    #     params["ontologies"] = ",".join(ontologies)

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except EnvironmentError as e:
        print(f"Error: {e}")
        return None

def extract_ontology_acronyms(data):
    """
    Extracts the number of ontologies and their acronyms from a JSON file.

    Parameters:
        json_file_path (str): Path to the JSON file.

    Returns:
        tuple: (total_count, list_of_acronyms)
    """
    # with open(json_file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    acronyms = []
    for item in data:
        ontologies = item.get("ontologies", [])
        for ontology in ontologies:
            acronym = ontology.get("acronym")
            if acronym:
                acronyms.append(acronym)

    return len(acronyms), acronyms
if __name__ == '__main__':
    example_text = "Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye."
    #annotated_relevant_classes = annotate_text(example_text)
    #print(json.dumps(annotated_relevant_classes, indent=2))
    example_text_2 = "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain"
    recommended_ontology_output = recommend_ontologies(example_text_2)
    top_ontologies = extract_ontology_acronyms(recommended_ontology_output)
    print(f"Input Text: {example_text_2}")
    pprint(top_ontologies)