"""
NER Ontology Coverage Analysis
==============================

This script analyzes the completeness of ontology mappings (IDs and labels) for
entities detected by each model. It identifies which models have better ontology
alignment capabilities.

Purpose: Analyze missing ontology IDs and labels per model
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


def analyze_ontology_coverage(loader: NERDataLoader, output_dir: Path):
    """
    Analyze ontology ID and label coverage for each model.
    
    Args:
        loader: NERDataLoader instance with loaded data
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model colors
    colors = loader.get_model_colors()
    
    # Collect statistics for all groups and models
    all_stats = {}
    
    for group in ['with_hil', 'without_hil']:
        if group not in loader.data or not loader.data[group]:
            continue
            
        group_stats = {}
        
        for model in loader.data[group].keys():
            # Get entity dataframe
            df = loader.get_entity_dataframe(group, model)
            
            if df.empty:
                continue
            
            # Get unique entities (deduplicate by entity text)
            unique_entities = df.groupby('entity').first().reset_index()
            
            # Calculate statistics
            total_entities = len(unique_entities)
            has_ontology_id = unique_entities['ontology_id'].notna() & (unique_entities['ontology_id'] != '')
            has_ontology_label = unique_entities['ontology_label'].notna() & (unique_entities['ontology_label'] != '')
            has_both = has_ontology_id & has_ontology_label
            has_neither = ~has_ontology_id & ~has_ontology_label
            has_id_only = has_ontology_id & ~has_ontology_label
            has_label_only = ~has_ontology_id & has_ontology_label
            
            group_stats[model] = {
                'total': total_entities,
                'has_both': has_both.sum(),
                'has_id_only': has_id_only.sum(),
                'has_label_only': has_label_only.sum(),
                'has_neither': has_neither.sum(),
                'has_any': (has_ontology_id | has_ontology_label).sum(),
                'missing_entities': unique_entities[has_neither]['entity'].tolist()
            }
        
        all_stats[group] = group_stats
        
        # Create visualization for this group
        create_ontology_coverage_plot(group_stats, group, colors, output_dir)
    
    # Create comparison between groups
    create_group_comparison_plot(all_stats, colors, output_dir)
    
    # Save detailed statistics
    save_ontology_statistics(all_stats, output_dir)


def create_ontology_coverage_plot(group_stats: Dict, group: str, colors: Dict[str, str], output_dir: Path):
    """
    Create stacked bar chart showing ontology coverage for a group.
    
    Args:
        group_stats: Statistics for all models in the group
        group: Group name
        colors: Model color mapping
        output_dir: Output directory
    """
    if not group_stats:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    
    # Order models: GPT, Claude, DeepSeek
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    models = [m for m in model_order if m in group_stats]
    n_models = len(models)
    
    # Prepare data for stacked bars
    has_both = [group_stats[m]['has_both'] for m in models]
    has_id_only = [group_stats[m]['has_id_only'] for m in models]
    has_label_only = [group_stats[m]['has_label_only'] for m in models]
    has_neither = [group_stats[m]['has_neither'] for m in models]
    
    # Create stacked bars
    x = np.arange(n_models)
    width = 0.6
    
    # Use different shades for the stacking
    p1 = ax.bar(x, has_both, width, label='Both ID & Label', 
                color='#2ca02c', alpha=0.9)
    p2 = ax.bar(x, has_id_only, width, bottom=has_both, 
                label='ID Only', color='#ff7f0e', alpha=0.9)
    p3 = ax.bar(x, has_label_only, width, 
                bottom=np.array(has_both) + np.array(has_id_only),
                label='Label Only', color='#1f77b4', alpha=0.9)
    p4 = ax.bar(x, has_neither, width,
                bottom=np.array(has_both) + np.array(has_id_only) + np.array(has_label_only),
                label='Neither', color='#d62728', alpha=0.9)
    
    # Customize plot
    ax.set_ylabel('Number of Entities', fontsize=7)
    ax.set_title(f'Ontology Coverage by Model ({group})', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax.legend(fontsize=6, loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add percentage labels
    # for i, model in enumerate(models):
    #     total = group_stats[model]['total']
    #     if total > 0:
    #         # Add percentage for "has_both" at the top of the bar
    #         pct_complete = (group_stats[model]['has_both'] / total) * 100
    #         max_total = max([group_stats[m]['total'] for m in models])
    #         ax.text(i, total + max_total * 0.05, f'{pct_complete:.0f}%', 
    #                ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'ontology_coverage_{group}.pdf', bbox_inches='tight')
    plt.close()
    

def create_group_comparison_plot(all_stats: Dict, colors: Dict[str, str], output_dir: Path):
    """
    Create comparison plot between with_hil and without_hil groups.
    
    Args:
        all_stats: Statistics for all groups and models
        colors: Model color mapping  
        output_dir: Output directory
    """
    if len(all_stats) < 2:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    
    # Get models that appear in both groups (ordered: GPT, Claude, DeepSeek)
    models_with = set(all_stats.get('with HIL', {}).keys())
    models_without = set(all_stats.get('without HIL', {}).keys())
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    common_models = [m for m in model_order if m in models_with and m in models_without]
    
    if not common_models:
        return
    
    # Calculate percentage of complete mappings for each model in each group
    x = np.arange(len(common_models))
    width = 0.35
    
    pct_with_hil = []
    pct_without_hil = []
    
    for model in common_models:
        # With HIL
        if 'with_hil' in all_stats and model in all_stats['with_hil']:
            stats = all_stats['with_hil'][model]
            pct = (stats['has_both'] / stats['total'] * 100) if stats['total'] > 0 else 0
            pct_with_hil.append(pct)
        else:
            pct_with_hil.append(0)
        
        # Without HIL
        if 'without_hil' in all_stats and model in all_stats['without_hil']:
            stats = all_stats['without_hil'][model]
            pct = (stats['has_both'] / stats['total'] * 100) if stats['total'] > 0 else 0
            pct_without_hil.append(pct)
        else:
            pct_without_hil.append(0)
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, pct_with_hil, width,
                   color=[colors.get(m, '#999999') for m in common_models], alpha=0.8)
    bars2 = ax.bar(x + width/2, pct_without_hil, width,
                   color=[colors.get(m, '#999999') for m in common_models], alpha=0.4)
    
    ax.set_ylabel('Complete Ontology Mappings (%)', fontsize=7)
    ax.set_title('Ontology Mapping: With vs Without HIL', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in common_models], fontsize=6)
    
    # # Add text labels instead of legend to avoid color confusion
    # ax.text(0.02, 0.95, 'With\nHIL', transform=ax.transAxes, fontsize=6, 
    #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # ax.text(0.12, 0.95, 'Without\nHIL', transform=ax.transAxes, fontsize=6, 
    #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'ontology_hil_comparison.pdf', bbox_inches='tight')
    plt.close()


def save_ontology_statistics(all_stats: Dict, output_dir: Path):
    """
    Save detailed ontology statistics to CSV files.
    
    Args:
        all_stats: Statistics for all groups and models
        output_dir: Output directory
    """
    # Create summary DataFrame
    summary_data = []
    
    for group, group_stats in all_stats.items():
        for model, stats in group_stats.items():
            total = stats['total']
            if total > 0:
                summary_data.append({
                    'Group': group,
                    'Model': model,
                    'Total Entities': total,
                    'Complete (ID & Label)': stats['has_both'],
                    'Complete %': f"{stats['has_both'] / total * 100:.1f}%",
                    'ID Only': stats['has_id_only'],
                    'Label Only': stats['has_label_only'],
                    'Neither': stats['has_neither'],
                    'Any Ontology Info': stats['has_any'],
                    'Any Ontology %': f"{stats['has_any'] / total * 100:.1f}%"
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'ontology_coverage_summary.csv', index=False)
    
    # Save entities missing ontology info
    missing_data = []
    
    for group, group_stats in all_stats.items():
        for model, stats in group_stats.items():
            for entity in stats['missing_entities']:
                missing_data.append({
                    'Group': group,
                    'Model': model,
                    'Entity': entity
                })
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        missing_df.to_csv(output_dir / 'entities_missing_ontology.csv', index=False)


def main():
    """Main function to run ontology coverage analysis."""
    # Initialize data loader
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Create output directory using absolute path from structsense root
    structsense_root = Path(__file__).parent.parent.parent
    output_dir = structsense_root / "evaluation/ner/evaluation/Latent-circuit/results"
    
    # Run analysis
    analyze_ontology_coverage(loader, output_dir)
    
    print(f"\nOntology coverage analysis complete. Results saved to {output_dir}")
    
    # Print summary
    for group in ['with_hil', 'without_hil']:
        if group in data and data[group]:
            print(f"\n{group.upper()} Ontology Coverage:")
            for model in data[group].keys():
                df = loader.get_entity_dataframe(group, model)
                if not df.empty:
                    unique_entities = df.groupby('entity').first().reset_index()
                    total = len(unique_entities)
                    has_both = (unique_entities['ontology_id'].notna() & 
                               (unique_entities['ontology_id'] != '') &
                               unique_entities['ontology_label'].notna() & 
                               (unique_entities['ontology_label'] != '')).sum()
                    pct = (has_both / total * 100) if total > 0 else 0
                    print(f"  {model}: {has_both}/{total} complete mappings ({pct:.1f}%)")


if __name__ == "__main__":
    main()