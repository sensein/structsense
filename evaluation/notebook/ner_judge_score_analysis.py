"""
NER Judge Score Analysis
========================

This script analyzes the judge scores assigned to entity extractions by each model.
It examines score distributions, confidence patterns, and quality assessments.

Purpose: Analyze judge score patterns and quality assessments
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import seaborn as sns

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


def analyze_judge_scores(loader: NERDataLoader, output_dir: Path):
    """
    Analyze judge scores across models and groups.
    
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
        
        # Collect score data for all models (ordered: GPT, Claude, DeepSeek)
        model_scores = {}
        
        model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
        for model in model_order:
            if model not in loader.data[group]:
                continue
                
            df = loader.get_entity_dataframe(group, model)
            
            if df.empty:
                continue
            
            # Filter out null scores and convert to numeric
            valid_scores = df[df['judge_score'].notna()]['judge_score'].astype(float)
            model_scores[model] = valid_scores.tolist()
        
        # Create score distribution visualizations
        create_score_density_plots(model_scores, group, colors, output_dir)
        create_score_statistics_table(model_scores, group, output_dir)
        
        # Analyze scores by entity characteristics
        analyze_scores_by_characteristics(loader, group, output_dir)
    
    # Create cross-group comparison
    create_group_score_comparison(loader, colors, output_dir)
    
    # Note: Removed correlation analysis as requested
    
    # Save detailed statistics
    save_score_statistics(loader, output_dir)


def create_score_density_plots(model_scores: Dict[str, List[float]], 
                              group: str, colors: Dict[str, str], output_dir: Path):
    """
    Create density plots showing score distributions for all models in one plot.
    
    Args:
        model_scores: Dictionary of model names to score lists
        group: Group name
        colors: Model color mapping
        output_dir: Output directory
    """
    if not model_scores:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    models = list(model_scores.keys())
    
    for model in models:
        scores = model_scores[model]
        if scores and len(scores) > 1:
            # Create density plot using histogram with many bins and normalization
            counts, bins = np.histogram(scores, bins=50, range=(0, 1), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Smooth the density curve
            from scipy import stats
            density = stats.gaussian_kde(scores)
            x_range = np.linspace(0, 1, 200)
            density_values = density(x_range)
            
            # Plot density curve
            ax.plot(x_range, density_values, color=colors.get(model, '#999999'), 
                   linewidth=2, alpha=0.8, label=model)
            
            # Fill under curve with transparency
            ax.fill_between(x_range, density_values, alpha=0.3, 
                           color=colors.get(model, '#999999'))
            
            # Add mean line
            mean_score = np.mean(scores)
            max_density = np.max(density_values)
            ax.axvline(mean_score, color=colors.get(model, '#999999'), 
                      linestyle='--', linewidth=1, alpha=0.7)
            
            # Add mean value text
            ax.text(mean_score, max_density * 1.05, f'{mean_score:.3f}', 
                   ha='center', va='bottom', fontsize=6, 
                   color=colors.get(model, '#999999'), fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Judge Score', fontsize=7)
    ax.set_ylabel('Density', fontsize=7)
    ax.set_title(f'Judge Score Distributions ({group})', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6, loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'judge_score_distributions_{group}.pdf', bbox_inches='tight')
    plt.close()


def create_score_statistics_table(model_scores: Dict[str, List[float]], 
                                group: str, output_dir: Path):
    """
    Create a table with detailed score statistics.
    
    Args:
        model_scores: Dictionary of model names to score lists
        group: Group name
        output_dir: Output directory
    """
    if not model_scores:
        return
    
    stats_data = []
    
    for model, scores in model_scores.items():
        if scores:
            stats = {
                'Model': model,
                'Count': len(scores),
                'Mean': np.mean(scores),
                'Median': np.median(scores),
                'Std': np.std(scores),
                'Min': np.min(scores),
                'Max': np.max(scores),
                'Q25': np.percentile(scores, 25),
                'Q75': np.percentile(scores, 75),
                'High_Confidence': sum(1 for s in scores if s >= 0.8) / len(scores) * 100,
                'Low_Confidence': sum(1 for s in scores if s < 0.5) / len(scores) * 100
            }
            stats_data.append(stats)
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Round numeric columns
        numeric_cols = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75', 
                       'High_Confidence', 'Low_Confidence']
        for col in numeric_cols:
            stats_df[col] = stats_df[col].round(3)
        
        # Save to CSV
        stats_df.to_csv(output_dir / f'judge_score_statistics_{group}.csv', index=False)
        
        # Create visual table
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = []
        headers = ['Model', 'Count', 'Mean', 'Median', 'Std', 'High Conf.%', 'Low Conf.%']
        
        for _, row in stats_df.iterrows():
            table_data.append([
                row['Model'].replace(' ', '\n'),
                int(row['Count']),
                f"{row['Mean']:.3f}",
                f"{row['Median']:.3f}",
                f"{row['Std']:.3f}",
                f"{row['High_Confidence']:.1f}%",
                f"{row['Low_Confidence']:.1f}%"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E6E6E6')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title(f'Judge Score Statistics Summary ({group})', 
                    fontsize=8, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_dir / f'judge_score_table_{group}.pdf', bbox_inches='tight')
        plt.close()


def analyze_scores_by_characteristics(loader: NERDataLoader, group: str, output_dir: Path):
    """
    Analyze judge scores by entity characteristics (label, location, etc.).
    
    Args:
        loader: NERDataLoader instance
        group: Group name
        output_dir: Output directory
    """
    if group not in loader.data or not loader.data[group]:
        return
    
    # Collect data across all models
    all_data = []
    
    for model in loader.data[group].keys():
        df = loader.get_entity_dataframe(group, model)
        
        if df.empty:
            continue
        
        # Filter valid scores
        valid_df = df[df['judge_score'].notna()].copy()
        valid_df['judge_score'] = valid_df['judge_score'].astype(float)
        valid_df['model'] = model
        
        all_data.append(valid_df)
    
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Analyze by label
    if 'label' in combined_df.columns:
        create_scores_by_label_plot(combined_df, group, output_dir)
    
    # Analyze by paper location
    if 'paper_location' in combined_df.columns:
        create_scores_by_location_plot(combined_df, group, output_dir)


def create_scores_by_label_plot(df: pd.DataFrame, group: str, output_dir: Path):
    """
    Create plot showing score distributions by entity label.
    
    Args:
        df: Combined dataframe with all models
        group: Group name
        output_dir: Output directory
    """
    # Get top labels by frequency
    top_labels = df['label'].value_counts().head(8).index.tolist()
    
    if len(top_labels) < 2:
        return
    
    # Filter to top labels
    filtered_df = df[df['label'].isin(top_labels)]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Create box plot
    label_data = [filtered_df[filtered_df['label'] == label]['judge_score'].tolist() 
                  for label in top_labels]
    
    bp = ax.boxplot(label_data, tick_labels=[label.replace('_', '\n') for label in top_labels],
                    patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f']
    
    for patch, color in zip(bp['boxes'], colors_cycle):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Judge Score', fontsize=7)
    ax.set_title(f'Judge Scores by Entity Label ({group})', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1)
    
    plt.xticks(fontsize=5)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'judge_scores_by_label_{group}.pdf', bbox_inches='tight')
    plt.close()


def create_scores_by_location_plot(df: pd.DataFrame, group: str, output_dir: Path):
    """
    Create plot showing score distributions by paper location.
    
    Args:
        df: Combined dataframe with all models
        group: Group name
        output_dir: Output directory
    """
    # Normalize locations (simple version)
    df['norm_location'] = df['paper_location'].str.lower().str.strip()
    
    # Get top locations by frequency
    top_locations = df['norm_location'].value_counts().head(6).index.tolist()
    
    if len(top_locations) < 2:
        return
    
    # Filter to top locations
    filtered_df = df[df['norm_location'].isin(top_locations)]
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    # Create box plot
    location_data = [filtered_df[filtered_df['norm_location'] == loc]['judge_score'].tolist() 
                    for loc in top_locations]
    
    bp = ax.boxplot(location_data, tick_labels=top_locations, patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for patch, color in zip(bp['boxes'], colors_cycle):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Judge Score', fontsize=7)
    ax.set_title(f'Judge Scores by Paper Location ({group})', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1)
    
    plt.xticks(fontsize=5)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'judge_scores_by_location_{group}.pdf', bbox_inches='tight')
    plt.close()


def create_group_score_comparison(loader: NERDataLoader, colors: Dict[str, str], output_dir: Path):
    """
    Create comparison of judge scores between groups.
    
    Args:
        loader: NERDataLoader instance
        colors: Model color mapping
        output_dir: Output directory
    """
    group_scores = {}
    
    for group in ['with_hil', 'without_hil']:
        if group not in loader.data or not loader.data[group]:
            continue
        
        model_scores = {}
        
        for model in loader.data[group].keys():
            df = loader.get_entity_dataframe(group, model)
            
            if not df.empty:
                valid_scores = df[df['judge_score'].notna()]['judge_score'].astype(float)
                model_scores[model] = valid_scores.tolist()
        
        group_scores[group] = model_scores
    
    if len(group_scores) < 2:
        return
    
    # Get models that appear in both groups (ordered: GPT, Claude, DeepSeek)
    models_with = set(group_scores.get('with_hil', {}).keys())
    models_without = set(group_scores.get('without_hil', {}).keys())
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    common_models = [m for m in model_order if m in models_with and m in models_without]
    
    if not common_models:
        return
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    x = np.arange(len(common_models))
    width = 0.35
    
    # Calculate mean scores
    means_with = []
    means_without = []
    
    for model in common_models:
        scores_with = group_scores['with_hil'].get(model, [])
        scores_without = group_scores['without_hil'].get(model, [])
        
        means_with.append(np.mean(scores_with) if scores_with else 0)
        means_without.append(np.mean(scores_without) if scores_without else 0)
    
    bars1 = ax.bar(x - width/2, means_with, width,
                   color=[colors.get(m, '#999999') for m in common_models], alpha=0.8)
    bars2 = ax.bar(x + width/2, means_without, width,
                   color=[colors.get(m, '#999999') for m in common_models], alpha=0.4)
    
    ax.set_ylabel('Mean Judge Score', fontsize=7)
    ax.set_title('Judge Scores: With vs Without HIL', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in common_models], fontsize=6)
    
    # Add text labels instead of legend to avoid color confusion
    ax.text(0.02, 0.95, 'With\nHIL', transform=ax.transAxes, fontsize=6, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.12, 0.95, 'Without\nHIL', transform=ax.transAxes, fontsize=6, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f'judge_scores_group_comparison.pdf', bbox_inches='tight')
    plt.close()




def save_score_statistics(loader: NERDataLoader, output_dir: Path):
    """
    Save detailed score statistics to CSV files.
    
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
            
            # Get valid scores
            valid_scores = df[df['judge_score'].notna()]['judge_score'].astype(float)
            
            if len(valid_scores) > 0:
                all_stats.append({
                    'Group': group,
                    'Model': model,
                    'Count': len(valid_scores),
                    'Mean': valid_scores.mean(),
                    'Median': valid_scores.median(),
                    'Std': valid_scores.std(),
                    'Min': valid_scores.min(),
                    'Max': valid_scores.max(),
                    'High_Confidence_Pct': (valid_scores >= 0.8).mean() * 100,
                    'Low_Confidence_Pct': (valid_scores < 0.5).mean() * 100
                })
    
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(output_dir / 'judge_score_detailed_statistics.csv', index=False)


def main():
    """Main function to run judge score analysis."""
    # Initialize data loader
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Create output directory using absolute path from structsense root
    structsense_root = Path(__file__).parent.parent.parent
    output_dir = structsense_root / "evaluation/ner/evaluation/Latent-circuit/results"
    
    # Run analysis
    analyze_judge_scores(loader, output_dir)
    
    print(f"\nJudge score analysis complete. Results saved to {output_dir}")
    
    # Print summary
    for group in ['with_hil', 'without_hil']:
        if group in data and data[group]:
            print(f"\n{group.upper()} Judge Score Summary:")
            for model in data[group].keys():
                df = loader.get_entity_dataframe(group, model)
                if not df.empty:
                    valid_scores = df[df['judge_score'].notna()]['judge_score'].astype(float)
                    if len(valid_scores) > 0:
                        mean_score = valid_scores.mean()
                        high_conf = (valid_scores >= 0.8).mean() * 100
                        print(f"  {model}: Mean={mean_score:.3f}, High confidence={high_conf:.1f}%")


if __name__ == "__main__":
    main()