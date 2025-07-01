"""
NER Comprehensive Analysis Summary
==================================

This script creates a comprehensive summary and comparison of all NER evaluation results,
combining insights from entity pool, ontology, location, label, and judge score analyses.

Author: Claude
Date: 2025-01-30
Purpose: Generate comprehensive summary report and final visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict

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


def generate_comprehensive_summary(loader: NERDataLoader, output_dir: Path):
    """
    Generate comprehensive summary analysis and report.
    
    Args:
        loader: NERDataLoader instance with loaded data
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model colors
    colors = loader.get_model_colors()
    
    # Collect comprehensive metrics
    summary_metrics = collect_all_metrics(loader)
    
    # Create comprehensive comparison dashboard
    create_performance_dashboard(summary_metrics, colors, output_dir)
    
    # Create detailed comparison tables
    create_summary_tables(summary_metrics, output_dir)
    
    # Generate text summary report
    generate_text_report(summary_metrics, output_dir)
    
    # Create model ranking visualization
    create_model_ranking(summary_metrics, colors, output_dir)


def collect_all_metrics(loader: NERDataLoader) -> Dict[str, Any]:
    """
    Collect all evaluation metrics from different analyses.
    
    Args:
        loader: NERDataLoader instance
        
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics = {
        'with_hil': {},
        'without_hil': {}
    }
    
    for group in ['with_hil', 'without_hil']:
        if group not in loader.data or not loader.data[group]:
            continue
        
        group_metrics = {}
        
        for model in loader.data[group].keys():
            df = loader.get_entity_dataframe(group, model)
            
            if df.empty:
                continue
            
            # Entity detection metrics
            overlap_stats = loader.calculate_entity_overlap(group)
            total_pool = overlap_stats['total_pool_size']
            detected = overlap_stats['model_counts'].get(model, 0)
            missed = overlap_stats['model_missing'].get(model, {}).get('count', 0)
            
            # Ontology coverage metrics
            unique_entities = df.groupby('entity').first().reset_index()
            total_entities = len(unique_entities)
            has_ontology_id = unique_entities['ontology_id'].notna() & (unique_entities['ontology_id'] != '')
            has_ontology_label = unique_entities['ontology_label'].notna() & (unique_entities['ontology_label'] != '')
            has_both = has_ontology_id & has_ontology_label
            
            # Judge score metrics
            valid_scores = df[df['judge_score'].notna()]['judge_score'].astype(float)
            
            # Label diversity metrics
            label_counts = unique_entities['label'].value_counts()
            shannon_diversity = 0
            if len(label_counts) > 0:
                for count in label_counts.values:
                    if count > 0:
                        p = count / total_entities
                        shannon_diversity -= p * np.log(p)
            
            # Paper location diversity
            location_counts = df['paper_location'].value_counts()
            
            group_metrics[model] = {
                # Entity detection
                'total_pool_size': total_pool,
                'entities_detected': detected,
                'entities_missed': missed,
                'detection_rate': (detected / total_pool * 100) if total_pool > 0 else 0,
                'miss_rate': (missed / total_pool * 100) if total_pool > 0 else 0,
                
                # Ontology coverage
                'total_unique_entities': total_entities,
                'ontology_complete': has_both.sum(),
                'ontology_complete_rate': (has_both.sum() / total_entities * 100) if total_entities > 0 else 0,
                'ontology_partial': (has_ontology_id | has_ontology_label).sum(),
                'ontology_partial_rate': ((has_ontology_id | has_ontology_label).sum() / total_entities * 100) if total_entities > 0 else 0,
                
                # Judge scores
                'judge_score_count': len(valid_scores),
                'judge_score_mean': valid_scores.mean() if len(valid_scores) > 0 else 0,
                'judge_score_std': valid_scores.std() if len(valid_scores) > 0 else 0,
                'high_confidence_rate': ((valid_scores >= 0.8).mean() * 100) if len(valid_scores) > 0 else 0,
                'low_confidence_rate': ((valid_scores < 0.5).mean() * 100) if len(valid_scores) > 0 else 0,
                
                # Label diversity
                'unique_labels': len(label_counts),
                'shannon_diversity': shannon_diversity,
                'most_common_label_pct': (label_counts.max() / total_entities * 100) if total_entities > 0 else 0,
                
                # Location diversity
                'unique_locations': len(location_counts),
                'total_occurrences': len(df)
            }
        
        metrics[group] = group_metrics
    
    return metrics


def create_performance_dashboard(metrics: Dict[str, Any], colors: Dict[str, str], output_dir: Path):
    """
    Create a performance dashboard with two radar plots and a legend in between.
    
    Args:
        metrics: Comprehensive metrics dictionary
        colors: Model color mapping
        output_dir: Output directory
    """
    # Create figure with two radar plots side by side
    fig = plt.figure(figsize=(10, 5))
    
    # Get common models across both groups (ordered: GPT, Claude, DeepSeek)
    models_with = set(metrics.get('with_hil', {}).keys())
    models_without = set(metrics.get('without_hil', {}).keys())
    model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
    common_models = [m for m in model_order if m in models_with and m in models_without]
    
    if not common_models:
        return
    
    # Create radar plot for with_hil
    ax1 = plt.subplot(131, projection='polar')
    create_radar_chart(ax1, metrics, 'with_hil', common_models, colors, 'With HIL')
    ax1.legend().remove()  # Remove individual legend
    
    # Create radar plot for without_hil
    ax3 = plt.subplot(133, projection='polar')
    create_radar_chart(ax3, metrics, 'without_hil', common_models, colors, 'Without HIL')
    ax3.legend().remove()  # Remove individual legend
    
    # Create central legend
    ax2 = plt.subplot(132)
    ax2.axis('off')  # Hide axes for legend area
    
    # Create legend handles
    handles = []
    for model in common_models:
        color = colors.get(model, '#999999')
        line = plt.Line2D([0], [0], color=color, linewidth=2, label=model)
        handles.append(line)
    
    # Place legend in the center
    legend = ax2.legend(handles=handles, loc='center', fontsize=10, 
                       frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure in PDF format only
    fig.savefig(output_dir / 'performance_dashboard.pdf', bbox_inches='tight')
    plt.close()


def create_metric_comparison(ax, metrics: Dict, models: List[str], metric_key: str, 
                           ylabel: str, colors: Dict[str, str]):
    """
    Create a comparison plot for a specific metric.
    
    Args:
        ax: Matplotlib axis
        metrics: Metrics dictionary
        models: List of models to compare
        metric_key: Key for the metric to plot
        ylabel: Y-axis label
        colors: Model color mapping
    """
    x = np.arange(len(models))
    width = 0.35
    
    values_with = [metrics.get('with_hil', {}).get(m, {}).get(metric_key, 0) for m in models]
    values_without = [metrics.get('without_hil', {}).get(m, {}).get(metric_key, 0) for m in models]
    
    bars1 = ax.bar(x - width/2, values_with, width,
                   color=[colors.get(m, '#999999') for m in models], alpha=0.8)
    bars2 = ax.bar(x + width/2, values_without, width,
                   color=[colors.get(m, '#999999') for m in models], alpha=0.4)
    
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values_with + values_without) * 0.01,
                       f'{height:.1f}' if metric_key != 'judge_score_mean' else f'{height:.3f}',
                       ha='center', va='bottom', fontsize=5)


def create_radar_chart(ax, metrics: Dict, group: str, models: List[str], 
                      colors: Dict[str, str], title: str):
    """
    Create a radar chart for model performance comparison.
    
    Args:
        ax: Matplotlib polar axis
        metrics: Metrics dictionary
        group: Group name
        models: List of models
        colors: Model color mapping
        title: Chart title
    """
    # Define performance dimensions (normalized to 0-1)
    dimensions = ['Detection\nRate', 'Ontology\nMapping', 'Judge\nScore', 
                 'Label\nDiversity', 'High\nConfidence']
    
    # Number of dimensions
    N = len(dimensions)
    
    # Compute angle for each dimension
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Plot each model
    for model in models:
        if model not in metrics.get(group, {}):
            continue
        
        model_data = metrics[group][model]
        
        # Normalize values to 0-1 scale
        values = [
            model_data.get('detection_rate', 0) / 100,  # 0-1
            model_data.get('ontology_complete_rate', 0) / 100,  # 0-1
            model_data.get('judge_score_mean', 0),  # already 0-1
            min(model_data.get('shannon_diversity', 0) / 3, 1),  # normalize to 0-1
            model_data.get('high_confidence_rate', 0) / 100  # 0-1
        ]
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=1, 
                label=model.replace(' ', '\n'), color=colors.get(model, '#999999'))
        ax.fill(angles, values, alpha=0.1, color=colors.get(model, '#999999'))
    
    # Customize the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=5)
    ax.grid(True)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=20)


def create_summary_tables(metrics: Dict[str, Any], output_dir: Path):
    """
    Create detailed summary tables with all metrics.
    
    Args:
        metrics: Comprehensive metrics dictionary
        output_dir: Output directory
    """
    # Create summary table data
    table_data = []
    
    for group, group_metrics in metrics.items():
        for model, model_metrics in group_metrics.items():
            table_data.append({
                'Group': group,
                'Model': model,
                'Entities_Detected': model_metrics.get('entities_detected', 0),
                'Detection_Rate_%': f"{model_metrics.get('detection_rate', 0):.1f}%",
                'Miss_Rate_%': f"{model_metrics.get('miss_rate', 0):.1f}%",
                'Ontology_Complete_%': f"{model_metrics.get('ontology_complete_rate', 0):.1f}%",
                'Judge_Score_Mean': f"{model_metrics.get('judge_score_mean', 0):.3f}",
                'High_Confidence_%': f"{model_metrics.get('high_confidence_rate', 0):.1f}%",
                'Label_Diversity': f"{model_metrics.get('shannon_diversity', 0):.2f}",
                'Unique_Labels': model_metrics.get('unique_labels', 0),
                'Unique_Locations': model_metrics.get('unique_locations', 0)
            })
    
    if table_data:
        summary_df = pd.DataFrame(table_data)
        summary_df.to_csv(output_dir / 'comprehensive_summary_table.csv', index=False)
        
        # Create performance ranking
        ranking_data = []
        
        for group in ['with_hil', 'without_hil']:
            if group not in metrics:
                continue
            
            group_ranking = []
            for model, model_metrics in metrics[group].items():
                # Calculate composite score (weighted average)
                composite_score = (
                    model_metrics.get('detection_rate', 0) * 0.3 +
                    model_metrics.get('ontology_complete_rate', 0) * 0.3 +
                    model_metrics.get('judge_score_mean', 0) * 100 * 0.2 +
                    min(model_metrics.get('shannon_diversity', 0) * 33.33, 100) * 0.1 +
                    model_metrics.get('high_confidence_rate', 0) * 0.1
                )
                
                group_ranking.append({
                    'Group': group,
                    'Model': model,
                    'Composite_Score': composite_score,
                    'Rank': 0  # Will be filled after sorting
                })
            
            # Sort and assign ranks
            group_ranking.sort(key=lambda x: x['Composite_Score'], reverse=True)
            for i, item in enumerate(group_ranking):
                item['Rank'] = i + 1
            
            ranking_data.extend(group_ranking)
        
        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df.to_csv(output_dir / 'model_rankings.csv', index=False)


def create_model_ranking(metrics: Dict[str, Any], colors: Dict[str, str], output_dir: Path):
    """
    Create visual model ranking comparison.
    
    Args:
        metrics: Comprehensive metrics dictionary
        colors: Model color mapping
        output_dir: Output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    for i, (group, ax) in enumerate([('with_hil', ax1), ('without_hil', ax2)]):
        if group not in metrics:
            continue
        
        # Calculate composite scores (use consistent ordering)
        model_scores = []
        
        model_order = ['GPT-4o-mini', 'Claude 3.7 Sonnet', 'DeepSeek V3 0324']
        for model in model_order:
            if model in metrics[group]:
                model_metrics = metrics[group][model]
                composite_score = (
                    model_metrics.get('detection_rate', 0) * 0.3 +
                    model_metrics.get('ontology_complete_rate', 0) * 0.3 +
                    model_metrics.get('judge_score_mean', 0) * 100 * 0.2 +
                    min(model_metrics.get('shannon_diversity', 0) * 33.33, 100) * 0.1 +
                    model_metrics.get('high_confidence_rate', 0) * 0.1
                )
                model_scores.append((model, composite_score))
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create bar chart
        models = [item[0] for item in model_scores]
        scores = [item[1] for item in model_scores]
        model_colors = [colors.get(m, '#999999') for m in models]
        
        bars = ax.barh(range(len(models)), scores, color=model_colors, alpha=0.8)
        
        # Customize plot
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.replace(' ', '\n') for m in models], fontsize=6)
        ax.set_xlabel('Composite Performance Score', fontsize=7)
        ax.set_title(f'Model Ranking ({group})', fontsize=8, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add score labels
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}', ha='left', va='center', fontsize=6)
        
        # Add rank numbers
        for j, (model, score) in enumerate(model_scores):
            ax.text(2, j, f'#{j+1}', ha='left', va='center', 
                   fontsize=6, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure in PDF format only
    fig.savefig(output_dir / 'model_rankings_visual.pdf', bbox_inches='tight')
    plt.close()


def generate_text_report(metrics: Dict[str, Any], output_dir: Path):
    """
    Generate a comprehensive text report summarizing all findings.
    
    Args:
        metrics: Comprehensive metrics dictionary
        output_dir: Output directory
    """
    report_lines = []
    
    report_lines.append("NER EVALUATION COMPREHENSIVE SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Overall findings
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 20)
    
    for group in ['with_hil', 'without_hil']:
        if group not in metrics:
            continue
        
        report_lines.append(f"\n{group.upper()} GROUP:")
        
        # Calculate group totals
        total_pool = 0
        best_detection = 0
        best_ontology = 0
        best_judge_score = 0
        best_model_detection = ""
        best_model_ontology = ""
        best_model_judge = ""
        
        for model, model_metrics in metrics[group].items():
            total_pool = model_metrics.get('total_pool_size', 0)
            
            detection_rate = model_metrics.get('detection_rate', 0)
            if detection_rate > best_detection:
                best_detection = detection_rate
                best_model_detection = model
            
            ontology_rate = model_metrics.get('ontology_complete_rate', 0)
            if ontology_rate > best_ontology:
                best_ontology = ontology_rate
                best_model_ontology = model
            
            judge_score = model_metrics.get('judge_score_mean', 0)
            if judge_score > best_judge_score:
                best_judge_score = judge_score
                best_model_judge = model
        
        report_lines.append(f"- Total unique entities in pool: {total_pool}")
        report_lines.append(f"- Best entity detection: {best_model_detection} ({best_detection:.1f}%)")
        report_lines.append(f"- Best ontology mapping: {best_model_ontology} ({best_ontology:.1f}%)")
        report_lines.append(f"- Highest judge scores: {best_model_judge} ({best_judge_score:.3f})")
    
    # Detailed model analysis
    report_lines.append("\n\nDETAILED MODEL ANALYSIS")
    report_lines.append("-" * 25)
    
    for group in ['with_hil', 'without_hil']:
        if group not in metrics:
            continue
        
        report_lines.append(f"\n{group.upper()} GROUP DETAILED METRICS:")
        report_lines.append("")
        
        for model, model_metrics in metrics[group].items():
            report_lines.append(f"{model}:")
            report_lines.append(f"  Entity Detection: {model_metrics.get('entities_detected', 0)}/{model_metrics.get('total_pool_size', 0)} ({model_metrics.get('detection_rate', 0):.1f}%)")
            report_lines.append(f"  Ontology Mapping: {model_metrics.get('ontology_complete', 0)}/{model_metrics.get('total_unique_entities', 0)} complete ({model_metrics.get('ontology_complete_rate', 0):.1f}%)")
            report_lines.append(f"  Judge Score: {model_metrics.get('judge_score_mean', 0):.3f} ± {model_metrics.get('judge_score_std', 0):.3f}")
            report_lines.append(f"  High Confidence: {model_metrics.get('high_confidence_rate', 0):.1f}%")
            report_lines.append(f"  Label Diversity: {model_metrics.get('unique_labels', 0)} types (Shannon: {model_metrics.get('shannon_diversity', 0):.2f})")
            report_lines.append("")
    
    # Cross-group comparison
    report_lines.append("\nCROSS-GROUP COMPARISON")
    report_lines.append("-" * 22)
    
    models_with = set(metrics.get('with_hil', {}).keys())
    models_without = set(metrics.get('without_hil', {}).keys())
    common_models = list(models_with & models_without)
    
    if common_models:
        report_lines.append("\nCommon models performance comparison:")
        report_lines.append("")
        
        for model in common_models:
            with_hil_metrics = metrics.get('with_hil', {}).get(model, {})
            without_hil_metrics = metrics.get('without_hil', {}).get(model, {})
            
            report_lines.append(f"{model}:")
            
            # Detection rate
            with_detection = with_hil_metrics.get('detection_rate', 0)
            without_detection = without_hil_metrics.get('detection_rate', 0)
            detection_diff = with_detection - without_detection
            report_lines.append(f"  Detection Rate: {with_detection:.1f}% (with HIL) vs {without_detection:.1f}% (without HIL) [Δ{detection_diff:+.1f}%]")
            
            # Ontology mapping
            with_ontology = with_hil_metrics.get('ontology_complete_rate', 0)
            without_ontology = without_hil_metrics.get('ontology_complete_rate', 0)
            ontology_diff = with_ontology - without_ontology
            report_lines.append(f"  Ontology Mapping: {with_ontology:.1f}% (with HIL) vs {without_ontology:.1f}% (without HIL) [Δ{ontology_diff:+.1f}%]")
            
            # Judge score
            with_judge = with_hil_metrics.get('judge_score_mean', 0)
            without_judge = without_hil_metrics.get('judge_score_mean', 0)
            judge_diff = with_judge - without_judge
            report_lines.append(f"  Judge Score: {with_judge:.3f} (with HIL) vs {without_judge:.3f} (without HIL) [Δ{judge_diff:+.3f}]")
            report_lines.append("")
    
    # Key insights
    report_lines.append("\nKEY INSIGHTS")
    report_lines.append("-" * 12)
    report_lines.append("")
    
    # Find overall best performer
    all_models = []
    for group, group_metrics in metrics.items():
        for model, model_metrics in group_metrics.items():
            composite_score = (
                model_metrics.get('detection_rate', 0) * 0.3 +
                model_metrics.get('ontology_complete_rate', 0) * 0.3 +
                model_metrics.get('judge_score_mean', 0) * 100 * 0.2 +
                min(model_metrics.get('shannon_diversity', 0) * 33.33, 100) * 0.1 +
                model_metrics.get('high_confidence_rate', 0) * 0.1
            )
            all_models.append((f"{model} ({group})", composite_score))
    
    all_models.sort(key=lambda x: x[1], reverse=True)
    
    if all_models:
        report_lines.append(f"1. Overall best performer: {all_models[0][0]} (composite score: {all_models[0][1]:.1f})")
    
    # HIL impact analysis
    if common_models:
        hil_improvements = 0
        hil_degradations = 0
        
        for model in common_models:
            with_hil_metrics = metrics.get('with_hil', {}).get(model, {})
            without_hil_metrics = metrics.get('without_hil', {}).get(model, {})
            
            # Compare detection rates
            detection_diff = with_hil_metrics.get('detection_rate', 0) - without_hil_metrics.get('detection_rate', 0)
            if detection_diff > 0:
                hil_improvements += 1
            elif detection_diff < 0:
                hil_degradations += 1
        
        if hil_improvements > hil_degradations:
            report_lines.append("2. Human-in-the-loop (HIL) generally improves performance")
        elif hil_degradations > hil_improvements:
            report_lines.append("2. Human-in-the-loop (HIL) appears to degrade performance in some cases")
        else:
            report_lines.append("2. Human-in-the-loop (HIL) shows mixed impact on performance")
    
    # Model-specific insights
    claude_performance = []
    gpt_performance = []
    deepseek_performance = []
    
    for group, group_metrics in metrics.items():
        for model, model_metrics in group_metrics.items():
            detection_rate = model_metrics.get('detection_rate', 0)
            if 'Claude' in model:
                claude_performance.append(detection_rate)
            elif 'GPT' in model:
                gpt_performance.append(detection_rate)
            elif 'DeepSeek' in model:
                deepseek_performance.append(detection_rate)
    
    avg_performances = []
    if claude_performance:
        avg_performances.append(('Claude', np.mean(claude_performance)))
    if gpt_performance:
        avg_performances.append(('GPT', np.mean(gpt_performance)))
    if deepseek_performance:
        avg_performances.append(('DeepSeek', np.mean(deepseek_performance)))
    
    if avg_performances:
        avg_performances.sort(key=lambda x: x[1], reverse=True)
        report_lines.append(f"3. Model ranking by average detection rate: {' > '.join([f'{model} ({rate:.1f}%)' for model, rate in avg_performances])}")
    
    # Save report
    report_text = '\n'.join(report_lines)
    
    with open(output_dir / 'comprehensive_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\nCOMPREHENSIVE ANALYSIS SUMMARY:")
    print("=" * 40)
    print(report_text)


def main():
    """Main function to run comprehensive summary analysis."""
    # Initialize data loader
    loader = NERDataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Create output directory using absolute path from structsense root
    structsense_root = Path(__file__).parent.parent.parent
    output_dir = structsense_root / "evaluation/ner/evaluation/Latent-circuit/results"
    
    # Run comprehensive analysis
    generate_comprehensive_summary(loader, output_dir)
    
    print(f"\nComprehensive summary analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()