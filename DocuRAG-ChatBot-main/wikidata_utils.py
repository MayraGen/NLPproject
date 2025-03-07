# wikidata_utils.py
import requests
from typing import Dict, List, Tuple, Optional

WIKITYPES = {"per": ["Q5"], "org": ["Q42"]}  # Person, Organization
SKIPPED_TYPES = ["tim"]  # Skip time entities

def fetch_wikidata(params: dict) -> requests.Response:
    """Basic Wikidata API request handler"""
    url = 'https://www.wikidata.org/w/api.php'
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"Error calling Wikidata API: {str(e)}")
        return None

def get_wikidata_entities(word_entity_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[dict]]:
    """Fetch Wikidata entities for given word-entity pairs"""
    wiki_entities = {}
    
    search_params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'uselang': 'en'
    }
    
    entity_params = {
        'action': 'wbgetentities',
        'format': 'json',
        'languages': 'en'
    }

    for word, label in word_entity_pairs:
        if label in SKIPPED_TYPES:
            continue
            
        search_params['search'] = word
        response = fetch_wikidata(search_params)
        
        if not response or 'search' not in response.json():
            continue
            
        data = response.json()
        
        # print(f"Wikidata entities corresponding to '{word}/{label}':")
        
        for result in data["search"]:
            description = result.get("description", "")
            identifier = result["id"]
            
            entity_params['ids'] = identifier
            entity_response = fetch_wikidata(entity_params)
            
            if not entity_response:
                continue
                
            entity_data = entity_response.json()
            
            for key in entity_data["entities"]:
                value = entity_data["entities"][key]
                if "P31" not in value["claims"]:
                    continue
                    
                entity_types = [typ["mainsnak"]["datavalue"]["value"]["id"] 
                              for typ in value["claims"]["P31"]]
                
                entity = {
                    "uri": result["concepturi"],
                    "text": result["match"]["text"],
                    "description": description,
                    "types": entity_types
                }
                
                if (word, label) not in wiki_entities:
                    wiki_entities[(word, label)] = []
                wiki_entities[(word, label)].append(entity)
                # print(f"\t{entity}")
                
    return wiki_entities

def get_simple_description(keyword: str) -> str:
    """Simplified description fetcher for basic use cases"""
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': keyword
    }
    
    response = fetch_wikidata(params)
    if response and response.json().get("search"):
        return response.json()["search"][0].get("description", "No description available")
    return "No description available"