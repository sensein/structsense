"""
NER Data Loader and Preprocessing Utilities
==========================================

This module provides utilities for loading and preprocessing NER evaluation data
from the Latent-circuit evaluation results. It handles both with_hil and without_hil
groups and standardizes the data structure for analysis.

Author: Claude
Date: 2025-01-30
Purpose: Load and preprocess NER evaluation JSON files for comparative analysis
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from collections import defaultdict
import sys

# Add structsense to Python path if needed
structsense_root = Path(__file__).parent.parent.parent
if str(structsense_root) not in sys.path:
    sys.path.insert(0, str(structsense_root))


class NERDataLoader:
    """
    Loads and preprocesses NER evaluation data from JSON files.
    
    Attributes:
        base_path (Path): Base path to the evaluation data
        model_mappings (Dict): Maps filename patterns to standardized model names
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Path to the Latent-circuit evaluation directory. 
                      If None, uses default path relative to structsense root.
        """
        if base_path is None:
            # Default path relative to structsense root
            self.base_path = structsense_root / "evaluation/ner/evaluation/Latent-circuit"
        else:
            self.base_path = Path(base_path)
            
        # Model mappings to match the token analysis script naming
        self.model_mappings = {
            'claude': 'Claude 3.7 Sonnet',
            'gpt': 'GPT-4o-mini',  # Note: it's 'gpt' not 'chatgpt' in filename
            'deepseek': 'DeepSeek V3 0324'
        }
        self.data = {'with_hil': {}, 'without_hil': {}}
        
    def load_all_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load all JSON files from both groups.
        
        Returns:
            Dictionary with structure: {group: {model: [entities]}}
        """
        for group in ['with_hil', 'without_hil']:
            group_path = self.base_path / group
            
            if not group_path.exists():
                print(f"Warning: {group_path} does not exist")
                continue
                
            # Find all JSON files in the directory
            json_files = list(group_path.glob("*.json"))
            
            for json_file in json_files:
                # Determine which model this file belongs to
                filename = json_file.name.lower()
                
                for pattern, model_name in self.model_mappings.items():
                    if pattern in filename:
                        print(f"Loading {json_file.name} as {model_name}")
                        self.data[group][model_name] = self._load_json_file(json_file, group)
                        break
                    
        return self.data
    
    def _load_json_file(self, filepath: Path, group: str) -> List[Dict]:
        """
        Load and flatten a single JSON file.
        
        Args:
            filepath: Path to the JSON file
            group: Group name (with_hil or without_hil)
            
        Returns:
            List of entity dictionaries
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures between groups
        entities = []
        
        if group == 'with_hil':
            # Structure: {judged_structured_information: {judge_ner_terms: {chunk_id: [entities]}}}
            if 'judged_structured_information' in data:
                ner_data = data['judged_structured_information'].get('judge_ner_terms', {})
            else:
                ner_data = {}
        else:
            # Structure: {judge_ner_terms: {judge_ner_terms: {chunk_id: [entities]}}}
            if 'judge_ner_terms' in data:
                ner_data = data['judge_ner_terms'].get('judge_ner_terms', {})
            else:
                ner_data = {}
        
        # Flatten all chunks into a single list
        for chunk_id, chunk_entities in ner_data.items():
            for entity in chunk_entities:
                # Store chunk number for reference if needed
                entity['_chunk_number'] = chunk_id
                entities.append(entity)
                
        return entities
    
    def get_entity_dataframe(self, group: str, model: str) -> pd.DataFrame:
        """
        Convert entity list to pandas DataFrame for easier analysis.
        
        Args:
            group: 'with_hil' or 'without_hil'
            model: Model name
            
        Returns:
            DataFrame with entity information (one row per occurrence)
        """
        if group not in self.data or model not in self.data[group]:
            return pd.DataFrame()
            
        entities = self.data[group][model]
        
        # Flatten nested lists for analysis
        flattened_entities = []
        for entity in entities:
            # Get the basic non-list fields
            entity_text = entity.get('entity', '')
            label = entity.get('label', '')
            ontology_id = entity.get('ontology_id', '')
            ontology_label = entity.get('ontology_label', '')
            chunk_number = entity.get('_chunk_number', '')
            
            # Get all list fields - these should all have the same length
            sentences = entity.get('sentence', [])
            starts = entity.get('start', [])
            ends = entity.get('end', [])
            remarks = entity.get('remarks', [])
            judge_scores = entity.get('judge_score', [])
            paper_locations = entity.get('paper_location', [])
            paper_titles = entity.get('paper_title', [])
            dois = entity.get('doi', [])
            
            # Create one row per occurrence
            num_occurrences = len(sentences) if sentences else 1
            
            for i in range(num_occurrences):
                row = {
                    'entity': entity_text,
                    'label': label,
                    'ontology_id': ontology_id,
                    'ontology_label': ontology_label,
                    'chunk_number': chunk_number,
                    'sentence': sentences[i] if i < len(sentences) else '',
                    'start': starts[i] if i < len(starts) else -1,
                    'end': ends[i] if i < len(ends) else -1,
                    'remark': remarks[i] if i < len(remarks) else '',
                    'judge_score': judge_scores[i] if i < len(judge_scores) else None,
                    'paper_location': paper_locations[i] if i < len(paper_locations) else '',
                    'paper_title': paper_titles[i] if i < len(paper_titles) else '',
                    'doi': dois[i] if i < len(dois) else ''
                }
                flattened_entities.append(row)
                
        return pd.DataFrame(flattened_entities)
    
    def get_unique_entities(self, group: str, model: str = None) -> List[str]:
        """
        Get unique entity strings for a group/model.
        
        Args:
            group: 'with_hil' or 'without_hil'
            model: Model name (if None, get all models in group)
            
        Returns:
            List of unique entity strings
        """
        entities = set()
        
        if model:
            models = [model] if model in self.data[group] else []
        else:
            models = self.data[group].keys()
            
        for m in models:
            for entity in self.data[group][m]:
                entities.add(entity.get('entity', '').lower().strip())
                
        return list(entities)
    
    def get_entity_pool(self, group: str) -> Dict[str, List[str]]:
        """
        Get the entity pool (union of all models) for a group.
        
        Args:
            group: 'with_hil' or 'without_hil'
            
        Returns:
            Dictionary with entity pool and per-model entities
        """
        pool = set()
        model_entities = {}
        
        for model in self.data[group].keys():
            entities = self.get_unique_entities(group, model)
            model_entities[model] = entities
            pool.update(entities)
                
        return {
            'pool': list(pool),
            'model_entities': model_entities
        }
    
    def calculate_entity_overlap(self, group: str) -> Dict[str, Any]:
        """
        Calculate entity overlap between models.
        
        Args:
            group: 'with_hil' or 'without_hil'
            
        Returns:
            Dictionary with overlap statistics
        """
        pool_data = self.get_entity_pool(group)
        pool = set(pool_data['pool'])
        model_entities = {m: set(e) for m, e in pool_data['model_entities'].items()}
        
        # Calculate overlaps
        overlap_stats = {
            'total_pool_size': len(pool),
            'model_counts': {m: len(e) for m, e in model_entities.items()},
            'model_missing': {},
            'pairwise_overlap': {},
            'all_models_shared': None
        }
        
        # Missing entities (false negatives) per model
        for model, entities in model_entities.items():
            missing = pool - entities
            overlap_stats['model_missing'][model] = {
                'count': len(missing),
                'entities': list(missing)
            }
        
        # Pairwise overlaps
        models = list(model_entities.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                overlap = model_entities[m1] & model_entities[m2]
                overlap_stats['pairwise_overlap'][f"{m1}_vs_{m2}"] = {
                    'count': len(overlap),
                    'entities': list(overlap)
                }
        
        # All models shared
        if len(model_entities) > 0:
            all_shared = set.intersection(*model_entities.values())
            overlap_stats['all_models_shared'] = {
                'count': len(all_shared),
                'entities': list(all_shared)
            }
            
        return overlap_stats
    
    def get_model_colors(self) -> Dict[str, str]:
        """
        Get the colorblind-friendly colors for each model.
        Matches the color scheme from token_cost_speed_analysis.py
        
        Returns:
            Dictionary mapping model names to hex colors
        """
        return {
            'GPT-4o-mini': '#E69F00',      # Orange
            'Claude 3.7 Sonnet': '#56B4E9', # Sky blue
            'DeepSeek V3 0324': '#009E73'   # Bluish green
        }


if __name__ == "__main__":
    # Example usage - no need to specify full path
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Print summary
    for group in ['with_hil', 'without_hil']:
        print(f"\n{group.upper()} Group:")
        for model, entities in data[group].items():
            print(f"  {model}: {len(entities)} entities")
            
    # Calculate overlap for with_hil group
    if 'with_hil' in data and data['with_hil']:
        overlap = loader.calculate_entity_overlap('with_hil')
        print(f"\nEntity pool size (with_hil): {overlap['total_pool_size']}")
        print("Model entity counts:", overlap['model_counts'])