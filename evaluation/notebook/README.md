# Token Cost Speed Analysis

A script to visualize token usage, cost, and speed metrics across different language models.

## Usage

```bash
python token_cost_speed_analysis.py --task <task_name> --file <csv_file>
```

### Arguments

- `--task`: Task name (e.g., `reproschema`). Creates output folder with this name.
- `--file`: Path to CSV file containing token usage data.

### Example

```bash
python token_cost_speed_analysis.py --task reproschema --file reproschema/reproschema_token_usage.csv
```

### Output

Generates 4 plots in both PNG and SVG formats:
- Cost distribution by model
- Token usage (input/output) by model  
- Speed distribution by model
- Speed vs cost scatter plot

All files are saved in a folder named after the task.