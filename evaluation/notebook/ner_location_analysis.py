"""
NER Paper Location Analysis
===========================

This script analyzes where in scientific papers each model detects entities.
It examines both overall detection patterns and location agreement for shared entities.

Purpose: Analyze paper location patterns for detected entities
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

# Add structsense to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.notebook.ner_data_loader import NERDataLoader

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3


def normalize_location(location: str) -> str:
    """
    Normalize paper location strings for consistency.
    
    Args:
        location: Raw location string
        
    Returns:
        Normalized location string or None if empty/whitespace
    """
    if not location or not location.strip():
        return None
    
    location = location.lower().strip()
    
    # Map variations to standard names
    location_map = {
        'introduction': 'introduction',
        'intro': 'introduction',
        'abstract': 'abstract',
        'results': 'results',
        'result': 'results',
        'methods': 'methods',
        'method': 'methods',
        'discussion': 'discussion',
        'conclusions': 'conclusion',
        'conclusion': 'conclusion',
        'references': 'references',
        'reference': 'references',
        'supplementary': 'supplementary',
        'supplement': 'supplementary',
        'figure': 'figure',
        'table': 'table',
        'introductory information': 'introduction',  # Map to introduction
        'author information': 'introduction',  # Map to introduction
        'acknowledgments': 'acknowledgments',
        'acknowledgements': 'acknowledgments'
    }
    
    # Check if location contains any of the keywords
    for key, value in location_map.items():
        if key in location:
            return value
    
    # If no match, return 'other'
    return 'other'


def analyze_paper_locations(loader: NERDataLoader, output_dir: Path):
    """
    Analyze paper location patterns for entity detection.
    
    Args:
        loader: NERDataLoader instance with loaded data
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model colors
    colors = loader.get_model_colors()
    
    # Analyze both groups
    for group in ['with_hil', 'without_hil']:
        if group not in loader.data or not loader.data[group]:
            continue
        
        # Collect location data for all models
        model_locations = {}
        all_locations = set()
        
        for model in loader.data[group].keys():
            df = loader.get_entity_dataframe(group, model)
            
            if df.empty:
                continue
            
            # Normalize locations and filter out None values
            df['normalized_location'] = df['paper_location'].apply(normalize_location)
            df_filtered = df[df['normalized_location'].notna()]
            
            # Count locations
            location_counts = df_filtered['normalized_location'].value_counts()
            model_locations[model] = location_counts
            all_locations.update(location_counts.index)
        
        # Create overall location distribution plot
        create_location_distribution_plot(model_locations, all_locations, group, colors, output_dir)
        
        # Create heatmap of entity detection by location
        create_location_heatmap(model_locations, group, output_dir)
        
    # Save detailed statistics
    save_location_statistics(loader, output_dir)


def create_location_distribution_plot(model_locations: Dict, all_locations: Set, 
                                    group: str, colors: Dict[str, str], output_dir: Path):
    """
    Create bar plot showing entity distribution across paper locations.
    
    Args:
        model_locations: Location counts per model
        all_locations: Set of all unique locations
        group: Group name
        colors: Model color mapping
        output_dir: Output directory
    """
    if not model_locations:
        return
    
    # Order locations as requested: abstract, introduction, results, discussion, methods
    location_order = ['abstract', 'introduction', 'results', 'discussion', 'methods', 
                     'conclusion', 'figure', 'table', 'supplementary', 'references', 
                     'acknowledgments', 'header', 'other']
    
    # Filter to only locations that exist
    locations = [loc for loc in location_order if loc in all_locations]
    
    # Prepare data (ordered: GPT, Claude, DeepSeek)
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    models = [m for m in model_order if m in model_locations]
    n_models = len(models)
    n_locations = len(locations)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Set up bar positions
    x = np.arange(n_locations)
    width = 0.8 / n_models
    
    # Plot bars for each model
    for i, model in enumerate(models):
        counts = [model_locations[model].get(loc, 0) for loc in locations]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width, 
                      label=model, color=colors.get(model, '#999999'), alpha=0.8)
        
        # Add value labels for bars > 0
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=5)
    
    ax.set_xlabel('Paper Location', fontsize=7)
    ax.set_ylabel('Number of Entity Occurrences', fontsize=7)
    ax.set_title(f'Entity Detection by Paper Location ({group})', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(locations, fontsize=6)
    ax.legend(fontsize=6, loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'location_distribution_{group}.pdf', bbox_inches='tight')
    plt.close()


def create_location_heatmap(model_locations: Dict, group: str, output_dir: Path):
    """
    Create heatmap showing entity detection patterns across locations.
    
    Args:
        model_locations: Location counts per model
        group: Group name
        output_dir: Output directory
    """
    if not model_locations:
        return
    
    # Prepare data for heatmap (ordered: GPT, Claude, DeepSeek)
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    models = [m for m in model_order if m in model_locations]
    location_order = ['abstract', 'introduction', 'results', 'discussion', 'methods', 
                     'conclusion', 'figure', 'table', 'supplementary', 'references', 
                     'acknowledgments', 'header', 'other']
    
    # Get all locations that appear in the data
    all_locations = set()
    for counts in model_locations.values():
        all_locations.update(counts.index)
    
    locations = [loc for loc in location_order if loc in all_locations]
    
    # Create matrix
    matrix = np.zeros((len(models), len(locations)))
    
    for i, model in enumerate(models):
        for j, location in enumerate(locations):
            matrix[i, j] = model_locations[model].get(location, 0)
    
    # Normalize by row (model) to show proportions
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.divide(matrix, row_sums, where=row_sums != 0) * 100
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    im = ax.imshow(matrix_normalized, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(locations)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(locations, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(models, fontsize=6)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(locations)):
            if matrix[i, j] > 0:  # Only show non-zero values
                text = ax.text(j, i, f'{matrix_normalized[i, j]:.0f}%',
                              ha="center", va="center", 
                              color="black" if matrix_normalized[i, j] < 50 else "white",
                              fontsize=5)
    
    ax.set_title(f'Entity Detection Heatmap by Location (%) - {group}', 
                fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Percentage of Entities (%)', fontsize=7)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'location_heatmap_{group}.pdf', bbox_inches='tight')
    plt.close()



def save_location_statistics(loader: NERDataLoader, output_dir: Path):
    """
    Save detailed location statistics to CSV files.
    
    Args:
        loader: NERDataLoader instance
        output_dir: Output directory
    """
    all_stats = []
    
    for group in ['with_hil', 'without_hil']:
        if group not in loader.data or not loader.data[group]:
            continue
        
        for model in loader.data[group].keys():
            df = loader.get_entity_dataframe(group, model)
            
            if df.empty:
                continue
            
            # Normalize locations and filter out None values
            df['normalized_location'] = df['paper_location'].apply(normalize_location)
            df_filtered = df[df['normalized_location'].notna()]
            
            # Get location counts
            location_counts = df_filtered['normalized_location'].value_counts()
            
            for location, count in location_counts.items():
                all_stats.append({
                    'Group': group,
                    'Model': model,
                    'Location': location,
                    'Count': count,
                    'Percentage': f"{count / len(df) * 100:.1f}%"
                })
    
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(output_dir / 'location_statistics.csv', index=False)


def main():
    """Main function to run paper location analysis."""
    # Initialize data loader
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Create output directory using absolute path from structsense root
    structsense_root = Path(__file__).parent.parent.parent
    output_dir = structsense_root / "evaluation/ner/evaluation/Latent-circuit/results"
    
    # Run analysis
    analyze_paper_locations(loader, output_dir)
    
    print(f"\nPaper location analysis complete. Results saved to {output_dir}")
    
    # Print summary
    for group in ['with_hil', 'without_hil']:
        if group in data and data[group]:
            print(f"\n{group.upper()} Location Summary:")
            for model in data[group].keys():
                df = loader.get_entity_dataframe(group, model)
                if not df.empty:
                    df['normalized_location'] = df['paper_location'].apply(normalize_location)
                    top_locations = df['normalized_location'].value_counts().head(3)
                    print(f"  {model} top locations: {', '.join([f'{loc} ({count})' for loc, count in top_locations.items()])}")


if __name__ == "__main__":
    main()