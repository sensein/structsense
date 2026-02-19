"""
NER Entity Pool Analysis
========================

This script analyzes the entity pool across models for both with_hil and without_hil groups.
It calculates false negatives (missed entities) for each model based on the union of all models.

Purpose: Analyze entity detection performance and calculate false negatives
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

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


def create_entity_pool_analysis(loader: NERDataLoader, group_name:Dict, output_dir: Path):
    """
    Create comprehensive entity pool analysis for both groups.
    
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
            print(f"No data for {group}, skipping...")
            continue
            
        # Calculate entity overlap statistics
        overlap_stats = loader.calculate_entity_overlap(group)
        
        # Create figure for entity counts and false negatives
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        
        # Plot 1: Entity counts per model (ordered: GPT, Claude, DeepSeek)
        model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
        models = [m for m in model_order if m in overlap_stats['model_counts']]
        counts = [overlap_stats['model_counts'][m] for m in models]
        model_colors = [colors.get(m, '#999999') for m in models]
        
        bars1 = ax1.bar(range(len(models)), counts, color=model_colors, alpha=0.8)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
        ax1.set_ylabel('Number of Unique Entities', fontsize=7)
        ax1.set_title(f'Entities Detected per Model ({group_name[group]})', fontsize=8, fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=6)
        
        # Add total pool size as horizontal line
        ax1.axhline(y=overlap_stats['total_pool_size'], color='red', 
                   linestyle='--', linewidth=1, alpha=0.5)
        max_count = max(counts) if counts else 0
        ax1.text(len(models)-0.5, overlap_stats['total_pool_size'] - max_count * 0.15, 
                f'Total Pool: {overlap_stats["total_pool_size"]}', 
                fontsize=6, ha='right', color='red')
        
        # Plot 2: Missed detection (missed entities)
        missed_counts = [overlap_stats['model_missing'][m]['count'] for m in models]
        
        bars2 = ax2.bar(range(len(models)), missed_counts, color=model_colors, alpha=0.8)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
        ax2.set_ylabel('Number of Missed Entities', fontsize=7)
        ax2.set_title(f'Missed Detection per Model ({group_name[group]})', fontsize=8, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_dir / f'entity_pool_analysis_{group}.pdf', bbox_inches='tight')
        plt.close()
        
        # Create Venn diagram for model overlaps (simplified version)
        create_overlap_visualization(overlap_stats, group, colors, group_name, output_dir)
        
        # Save detailed statistics to CSV
        save_overlap_statistics(overlap_stats, group, output_dir)


def create_overlap_visualization(overlap_stats: Dict, group: str, colors: Dict[str, str], group_name: Dict, output_dir: Path):
    """
    Create visualization of entity overlaps between models.
    
    Args:
        overlap_stats: Dictionary with overlap statistics
        group: Group name
        colors: Model color mapping
        output_dir: Output directory
    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    
    # Prepare data for visualization (ordered: GPT, Claude, DeepSeek)
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    models = [m for m in model_order if m in overlap_stats['model_counts']]
    n_models = len(models)
    
    # Create a matrix of overlaps
    overlap_matrix = np.zeros((n_models, n_models))
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                overlap_matrix[i, j] = overlap_stats['model_counts'][m1]
            else:
                key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in overlap_stats['pairwise_overlap'] else f"{m2}_vs_{m1}"
                if key in overlap_stats['pairwise_overlap']:
                    overlap_matrix[i, j] = overlap_stats['pairwise_overlap'][key]['count']
    
    # Plot heatmap
    im = ax.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax.set_yticklabels(models, fontsize=6)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            text = ax.text(j, i, int(overlap_matrix[i, j]),
                          ha="center", va="center", color="black" if overlap_matrix[i, j] < 20 else "white",
                          fontsize=6)
    
    ax.set_title(f'Entity Overlap Matrix ({group_name[group]})', fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Number of Entities', fontsize=7)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'entity_overlap_matrix_{group}.pdf', bbox_inches='tight')
    plt.close()
    
    # Also create a bar chart showing shared entities
    if overlap_stats['all_models_shared']:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        
        categories = ['All Models', 'Any Two Models', 'Model-Specific']
        values = [
            overlap_stats['all_models_shared']['count'],
            # Average pairwise overlap
            np.mean([v['count'] for v in overlap_stats['pairwise_overlap'].values()]),
            # Average unique to each model (estimated)
            np.mean([overlap_stats['model_counts'][m] - overlap_stats['all_models_shared']['count'] 
                    for m in models if m in overlap_stats['model_counts']])
        ]
        
        bars = ax.bar(categories, values, color=['#66c2a5', '#fc8d62', '#8da0cb'], alpha=0.8)
        ax.set_ylabel('Number of Entities', fontsize=7)
        ax.set_title(f'Entity Sharing Patterns ({group_name[group]})', fontsize=8, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=6)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_dir / f'entity_sharing_patterns_{group}.pdf', bbox_inches='tight')
        plt.close()


def save_overlap_statistics(overlap_stats: Dict, group: str, output_dir: Path):
    """
    Save detailed overlap statistics to CSV files.
    
    Args:
        overlap_stats: Dictionary with overlap statistics
        group: Group name
        output_dir: Output directory
    """
    # Summary statistics
    summary_data = {
        'Metric': ['Total Pool Size', 'All Models Shared'],
        'Value': [overlap_stats['total_pool_size'], 
                 overlap_stats['all_models_shared']['count'] if overlap_stats['all_models_shared'] else 0]
    }
    
    # Add model-specific stats
    for model, count in overlap_stats['model_counts'].items():
        summary_data['Metric'].extend([f'{model} - Detected', f'{model} - Missed'])
        summary_data['Value'].extend([count, overlap_stats['model_missing'][model]['count']])
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / f'entity_pool_summary_{group}.csv', index=False)
    
    # Detailed missed entities
    missed_entities_data = []
    for model, data in overlap_stats['model_missing'].items():
        for entity in data['entities']:
            missed_entities_data.append({
                'Model': model,
                'Missed Entity': entity
            })
    
    if missed_entities_data:
        missed_df = pd.DataFrame(missed_entities_data)
        missed_df.to_csv(output_dir / f'missed_entities_details_{group}.csv', index=False)
    
    # Shared entities details
    if overlap_stats['all_models_shared'] and overlap_stats['all_models_shared']['entities']:
        shared_df = pd.DataFrame({
            'Shared Entity': overlap_stats['all_models_shared']['entities']
        })
        shared_df.to_csv(output_dir / f'shared_entities_all_models_{group}.csv', index=False)


def main():
    """Main function to run entity pool analysis."""
    # Initialize data loader
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Create output directory using absolute path from structsense root
    structsense_root = Path(__file__).parent.parent.parent
    output_dir = structsense_root / "evaluation/ner/evaluation/Latent-circuit/results"
    
    group_name = {"with_hil": "with HIL", "without_hil": "without HIL"}

    # Run analysis
    create_entity_pool_analysis(loader, group_name, output_dir)
    
    print(f"\nEntity pool analysis complete. Results saved to {output_dir}")
    
    # Print summary
    for group in ['with_hil', 'without_hil']:
        if group in data and data[group]:
            overlap = loader.calculate_entity_overlap(group)
            print(f"\n{group.upper()} Summary:")
            print(f"  Total entity pool: {overlap['total_pool_size']}")
            print(f"  Entities shared by all models: {overlap['all_models_shared']['count'] if overlap['all_models_shared'] else 0}")
            for model, count in overlap['model_counts'].items():
                miss_rate = overlap['model_missing'][model]['count'] / overlap['total_pool_size'] * 100
                print(f"  {model}: {count} detected, {overlap['model_missing'][model]['count']} missed ({miss_rate:.1f}% miss rate)")


if __name__ == "__main__":
    main()