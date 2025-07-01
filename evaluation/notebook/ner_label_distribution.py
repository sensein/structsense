"""
NER Label Distribution Analysis
===============================

This script analyzes the distribution of entity labels (types) detected by each model.
It examines both individual model patterns and shared entity labels across models.

Purpose: Analyze entity label distribution patterns
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


def analyze_label_distributions(loader: NERDataLoader, output_dir: Path):
    """
    Analyze entity label distributions across models and groups.
    
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
        
        # Collect label data for all models (ordered: GPT, Claude, DeepSeek)
        model_labels = {}
        all_labels = set()
        
        model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
        for model in model_order:
            if model not in loader.data[group]:
                continue
                
            df = loader.get_entity_dataframe(group, model)
            
            if df.empty:
                continue
            
            # Get unique entities with their labels
            unique_entities = df.groupby('entity').first().reset_index()
            
            # Count labels
            label_counts = unique_entities['label'].value_counts()
            model_labels[model] = label_counts
            all_labels.update(label_counts.index)
        
        # Create label distribution visualizations
        create_label_distribution_plot(model_labels, all_labels, group, colors, output_dir)
        create_label_diversity_plot(model_labels, group, colors, output_dir)
        
        # Analyze shared entity labels
        analyze_shared_entity_labels(loader, group, output_dir)
    
    # Create cross-group comparison
    create_group_label_comparison(loader, colors, output_dir)
    
    # Save detailed statistics
    save_label_statistics(loader, output_dir)


def create_label_distribution_plot(model_labels: Dict, all_labels: Set, 
                                 group: str, colors: Dict[str, str], output_dir: Path):
    """
    Create horizontal bar plot showing label distribution with models as grouped bars.
    
    Args:
        model_labels: Label counts per model
        all_labels: Set of all unique labels
        group: Group name
        colors: Model color mapping
        output_dir: Output directory
    """
    if not model_labels:
        return
    
    # Sort labels by total frequency across all models
    label_totals = defaultdict(int)
    for counts in model_labels.values():
        for label, count in counts.items():
            label_totals[label] += count
    
    sorted_labels = sorted(label_totals.keys(), key=lambda x: label_totals[x], reverse=True)
    
    # Create single horizontal bar plot with models grouped
    fig, ax = plt.subplots(1, 1, figsize=(6, max(4, len(sorted_labels) * 0.4)))
    
    models = list(model_labels.keys())
    n_models = len(models)
    n_labels = len(sorted_labels)
    
    # Set up bar positions
    y_pos = np.arange(n_labels)
    bar_height = 0.8 / n_models
    
    # Plot bars for each model
    for i, model in enumerate(models):
        counts = [model_labels[model].get(label, 0) for label in sorted_labels]
        offset = (i - n_models/2 + 0.5) * bar_height
        
        bars = ax.barh(y_pos + offset, counts, bar_height, 
                      label=model, color=colors.get(model, '#999999'), alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=6)
    ax.set_xlabel('Number of Entities', fontsize=7)
    ax.set_title(f'Entity Label Distribution ({group})', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend in bottom right
    ax.legend(fontsize=6, loc='lower right', frameon=False)
    
    # Set integer ticks on x-axis
    max_count = max([counts.max() if len(counts) > 0 else 0 for counts in model_labels.values()])
    if max_count > 0:
        ax.set_xticks(range(0, int(max_count) + 1, max(1, int(max_count) // 5)))
    
    # Reverse y-axis to show most frequent at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'label_distribution_{group}.pdf', bbox_inches='tight')
    plt.close()


def create_label_diversity_plot(model_labels: Dict, group: str, colors: Dict[str, str], output_dir: Path):
    """
    Create plot showing label diversity metrics per model.
    
    Args:
        model_labels: Label counts per model
        group: Group name
        colors: Model color mapping
        output_dir: Output directory
    """
    if not model_labels:
        return
    
    # Use consistent model ordering
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    models = [m for m in model_order if m in model_labels]
    
    # Calculate diversity metrics
    diversity_metrics = {}
    
    for model, label_counts in model_labels.items():
        total_entities = label_counts.sum()
        n_unique_labels = len(label_counts)
        
        # Shannon diversity index
        shannon_diversity = 0
        for count in label_counts.values:
            if count > 0:
                p = count / total_entities
                shannon_diversity -= p * np.log(p)
        
        # Simpson diversity index (1 - Simpson's dominance)
        simpson_diversity = 1 - sum((count / total_entities) ** 2 for count in label_counts.values)
        
        # Most common label percentage
        max_label_pct = label_counts.max() / total_entities * 100 if total_entities > 0 else 0
        
        diversity_metrics[model] = {
            'n_unique_labels': n_unique_labels,
            'shannon_diversity': shannon_diversity,
            'simpson_diversity': simpson_diversity,
            'max_label_pct': max_label_pct,
            'total_entities': total_entities
        }
    
    # Create diversity comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 5))
    
    x = np.arange(len(models))
    model_colors = [colors.get(m, '#999999') for m in models]
    
    # Plot 1: Number of unique labels
    unique_counts = [diversity_metrics[m]['n_unique_labels'] for m in models]
    bars1 = ax1.bar(x, unique_counts, color=model_colors, alpha=0.8)
    ax1.set_ylabel('Number of Unique Labels', fontsize=7)
    ax1.set_title('Label Type Diversity', fontsize=8, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=6)
    
    # Plot 2: Shannon diversity
    shannon_scores = [diversity_metrics[m]['shannon_diversity'] for m in models]
    bars2 = ax2.bar(x, shannon_scores, color=model_colors, alpha=0.8)
    ax2.set_ylabel('Shannon Diversity Index', fontsize=7)
    ax2.set_title('Shannon Diversity', fontsize=8, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)
    
    # Plot 3: Simpson diversity
    simpson_scores = [diversity_metrics[m]['simpson_diversity'] for m in models]
    bars3 = ax3.bar(x, simpson_scores, color=model_colors, alpha=0.8)
    ax3.set_ylabel('Simpson Diversity Index', fontsize=7)
    ax3.set_title('Simpson Diversity', fontsize=8, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=6)
    
    # Plot 4: Dominance (most common label percentage)
    dominance_scores = [diversity_metrics[m]['max_label_pct'] for m in models]
    bars4 = ax4.bar(x, dominance_scores, color=model_colors, alpha=0.8)
    ax4.set_ylabel('Most Common Label (%)', fontsize=7)
    ax4.set_title('Label Dominance', fontsize=8, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=6)
    
    plt.suptitle(f'Label Diversity Metrics ({group})', fontsize=9, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'label_diversity_{group}.pdf', bbox_inches='tight')
    plt.close()


def analyze_shared_entity_labels(loader: NERDataLoader, group: str, output_dir: Path):
    """
    Analyze labels of entities detected by multiple models.
    
    Args:
        loader: NERDataLoader instance
        group: Group name
        output_dir: Output directory
    """
    if group not in loader.data or not loader.data[group]:
        return
    
    # Get shared entities
    overlap_stats = loader.calculate_entity_overlap(group)
    
    # Analyze pairwise overlaps
    if overlap_stats['pairwise_overlap']:
        label_agreements = []
        
        for pair_key, pair_data in overlap_stats['pairwise_overlap'].items():
            if pair_data['count'] == 0:
                continue
                
            model1, model2 = pair_key.split('_vs_')
            shared_entities = set(pair_data['entities'])
            
            # Get labels for these entities from both models
            for model in [model1, model2]:
                df = loader.get_entity_dataframe(group, model)
                if df.empty:
                    continue
                
                # Filter to shared entities
                shared_df = df[df['entity'].str.lower().str.strip().isin(shared_entities)]
                entity_labels = shared_df.groupby('entity')['label'].first()
                
                for entity, labels in entity_labels.items():
                    label_agreements.append({
                        'pair': pair_key,
                        'model': model,
                        'entity': entity.lower().strip(),
                        'label': labels
                    })
        
        if label_agreements:
            # Create label agreement analysis
            agreement_df = pd.DataFrame(label_agreements)
            
            # Calculate label consistency for shared entities
            label_consistency = []
            
            for pair_key in agreement_df['pair'].unique():
                pair_df = agreement_df[agreement_df['pair'] == pair_key]
                
                for entity in pair_df['entity'].unique():
                    entity_df = pair_df[pair_df['entity'] == entity]
                    if len(entity_df) == 2:  # Both models have this entity
                        labels = entity_df['label'].tolist()
                        consistent = len(set(labels)) == 1
                        label_consistency.append({
                            'pair': pair_key,
                            'entity': entity,
                            'consistent': consistent,
                            'labels': ', '.join(labels)
                        })
            
            if label_consistency:
                consistency_df = pd.DataFrame(label_consistency)
                
                # Create consistency visualization
                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                
                consistency_stats = consistency_df.groupby('pair')['consistent'].agg(['sum', 'count'])
                consistency_stats['inconsistent'] = consistency_stats['count'] - consistency_stats['sum']
                consistency_stats['consistent_pct'] = (consistency_stats['sum'] / consistency_stats['count'] * 100)
                
                pairs = consistency_stats.index.tolist()
                consistent_counts = consistency_stats['sum'].tolist()
                inconsistent_counts = consistency_stats['inconsistent'].tolist()
                
                x = np.arange(len(pairs))
                width = 0.6
                
                bars1 = ax.bar(x, consistent_counts, width, label='Consistent Labels',
                              color='#2ca02c', alpha=0.8)
                bars2 = ax.bar(x, inconsistent_counts, width, bottom=consistent_counts,
                              label='Inconsistent Labels', color='#d62728', alpha=0.8)
                
                ax.set_ylabel('Number of Shared Entities', fontsize=7)
                ax.set_title(f'Label Consistency for Shared Entities ({group})', 
                            fontsize=8, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([p.replace('_vs_', '\nvs\n') for p in pairs], fontsize=6)
                ax.legend(fontsize=6, loc='upper right', frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add percentage labels
                for i, pct in enumerate(consistency_stats['consistent_pct']):
                    total_height = consistency_stats.iloc[i]['count']
                    max_height = max(consistency_stats['count'])
                    ax.text(i, total_height + max_height * 0.05,
                           f'{pct:.0f}%', ha='center', va='bottom', fontsize=6)
                
                plt.tight_layout()
                
                # Save figure
                fig.savefig(output_dir / f'label_consistency_{group}.pdf', bbox_inches='tight')
                plt.close()


def create_group_label_comparison(loader: NERDataLoader, colors: Dict[str, str], output_dir: Path):
    """
    Create comparison of label distributions between groups.
    
    Args:
        loader: NERDataLoader instance
        colors: Model color mapping
        output_dir: Output directory
    """
    group_stats = {}
    
    for group in ['with_hil', 'without_hil']:
        if group not in loader.data or not loader.data[group]:
            continue
        
        group_labels = defaultdict(int)
        
        for model in loader.data[group].keys():
            df = loader.get_entity_dataframe(group, model)
            if not df.empty:
                unique_entities = df.groupby('entity').first().reset_index()
                label_counts = unique_entities['label'].value_counts()
                
                for label, count in label_counts.items():
                    group_labels[label] += count
        
        group_stats[group] = dict(group_labels)
    
    if len(group_stats) < 2:
        return
    
    # Get all labels
    all_labels = set()
    for labels in group_stats.values():
        all_labels.update(labels.keys())
    
    # Sort by total frequency
    label_totals = defaultdict(int)
    for labels in group_stats.values():
        for label, count in labels.items():
            label_totals[label] += count
    
    sorted_labels = sorted(label_totals.keys(), key=lambda x: label_totals[x], reverse=True)
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    x = np.arange(len(sorted_labels))
    width = 0.35
    
    with_hil_counts = [group_stats.get('with_hil', {}).get(label, 0) for label in sorted_labels]
    without_hil_counts = [group_stats.get('without_hil', {}).get(label, 0) for label in sorted_labels]
    
    bars1 = ax.bar(x - width/2, with_hil_counts, width, label='With HIL',
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, without_hil_counts, width, label='Without HIL',
                   color='#ff7f0e', alpha=0.8)
    
    ax.set_ylabel('Total Number of Entities', fontsize=7)
    ax.set_title('Entity Label Distribution: With vs Without HIL', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('_', '\n') for label in sorted_labels], 
                       rotation=90, ha='center', fontsize=5.5)
    ax.legend(fontsize=6, loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust margins to fit rotated labels
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'label_distribution_group_comparison.pdf', bbox_inches='tight')
    plt.close()


def save_label_statistics(loader: NERDataLoader, output_dir: Path):
    """
    Save detailed label statistics to CSV files.
    
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
            
            # Get unique entities with their labels
            unique_entities = df.groupby('entity').first().reset_index()
            label_counts = unique_entities['label'].value_counts()
            
            for label, count in label_counts.items():
                all_stats.append({
                    'Group': group,
                    'Model': model,
                    'Label': label,
                    'Count': count,
                    'Percentage': f"{count / len(unique_entities) * 100:.1f}%"
                })
    
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(output_dir / 'label_distribution_statistics.csv', index=False)


def main():
    """Main function to run label distribution analysis."""
    # Initialize data loader
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Create output directory using absolute path from structsense root
    structsense_root = Path(__file__).parent.parent.parent
    output_dir = structsense_root / "evaluation/ner/evaluation/Latent-circuit/results"
    
    # Run analysis
    analyze_label_distributions(loader, output_dir)
    
    print(f"\nLabel distribution analysis complete. Results saved to {output_dir}")
    
    # Print summary
    for group in ['with_hil', 'without_hil']:
        if group in data and data[group]:
            print(f"\n{group.upper()} Label Summary:")
            for model in data[group].keys():
                df = loader.get_entity_dataframe(group, model)
                if not df.empty:
                    unique_entities = df.groupby('entity').first().reset_index()
                    top_labels = unique_entities['label'].value_counts().head(3)
                    print(f"  {model} top labels: {', '.join([f'{label} ({count})' for label, count in top_labels.items()])}")


if __name__ == "__main__":
    main()