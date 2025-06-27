import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate token usage and cost analysis plots')
parser.add_argument('--task', type=str, required=True, help='Task name (e.g., reproschema)')
parser.add_argument('--file', type=str, required=True, help='CSV file name (e.g., reproschema_token_usage.csv)')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = args.task
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Read data
df = pd.read_csv(args.file)
df['Model'] = df['Model'].str.strip()

# Define colorblind-friendly palette
# Using colors from Wong's colorblind-friendly palette
model_colors = {
    'GPT-4o-mini': '#E69F00',  # Orange
    'Claude 3.7 Sonnet': '#56B4E9',  # Sky blue
    'DeepSeek V3 0324': '#009E73'  # Bluish green
}

# Token type colors (different from model colors)
token_colors = {
    'Input Tokens': '#CC79A7',  # Reddish purple
    'Output Tokens': '#F0E442'  # Yellow
}

# Set matplotlib parameters for small figures
plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 8,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Plot 1: Cost per Model (Violin Plot)
plt.figure(figsize=(3.2, 2.5))
ax = sns.violinplot(data=df, x='Model', y='Cost ($)', hue='Model', inner='box', 
                    palette=model_colors, linewidth=0.8, legend=False)
plt.title("Cost Distribution by Model", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{args.task}_cost_violin.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f"{args.task}_cost_violin.svg"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Input and Output Tokens per Model (Violin Plot)
plt.figure(figsize=(3.5, 3))
df_long = df.melt(id_vars=['Model'], value_vars=['Input Tokens', 'Output Tokens'],
                  var_name='Token Type', value_name='Count')

# Create custom palette for split violin
split_palette = []
for model in df['Model'].unique():
    split_palette.extend([token_colors['Input Tokens'], token_colors['Output Tokens']])

ax = sns.violinplot(data=df_long, x='Model', y='Count', hue='Token Type', 
                    split=True, palette=token_colors, linewidth=0.8)
ax.set_xticklabels(ax.get_xticklabels())
plt.title("Token Usage by Model", pad=10)
plt.legend(title='Token Type', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{args.task}_token_usage_violin.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f"{args.task}_token_usage_violin.svg"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Speed per Model (Violin Plot)
plt.figure(figsize=(3.2, 2.5))
ax = sns.violinplot(data=df, x='Model', y='Speed (tps)', hue='Model', inner='box', 
                    palette=model_colors, linewidth=0.8, legend=False)
plt.title("Speed Distribution by Model", pad=10)
plt.ylabel("Speed (tokens/sec)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{args.task}_speed_violin.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f"{args.task}_speed_violin.svg"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Scatter plot of Speed vs Cost
plt.figure(figsize=(3, 3))
markers = ['o', 's', '^']  # Different markers for each model
for i, (model, group) in enumerate(df.groupby('Model')):
    plt.scatter(group['Speed (tps)'], group['Cost ($)'], 
                color=model_colors[model], label=model, 
                marker=markers[i], s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
plt.xlabel("Speed (tokens/sec)")
plt.ylabel("Cost ($)")
plt.title("Speed vs Cost by Model", pad=10)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{args.task}_speed_vs_cost.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f"{args.task}_speed_vs_cost.svg"), dpi=300, bbox_inches='tight')
plt.show()