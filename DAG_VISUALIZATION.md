# DAG Visualization Guide

This guide explains how to use the DAG visualization tool in the CPC DAG Analyzer project.

## Overview

The DAG visualization tool provides a visual representation of your task graph, making it easier to understand dependencies, node types, execution times, and critical paths.

## Features

### 1. **Multiple Layout Algorithms**
- **Hierarchical** (default): Arranges nodes in levels based on dependencies
- **Spring**: Uses force-directed layout for natural clustering
- **Circular**: Arranges nodes in a circular pattern

### 2. **Color-Coded Node Types**
- **Sky Blue**: Default nodes (regular tasks)
- **Pink/Salmon**: Provider nodes (different shades for different provider groups)
- **Gold**: Self-looping nodes (iterative algorithms)

### 3. **Critical Path Highlighting**
When running with the CPC analyzer, critical path edges are highlighted in red with thicker lines.

### 4. **Comprehensive Node Information**
Each node displays:
- Shortened node name (for readability)
- Execution time in milliseconds
- Type indicator ([PROV] for provider, [LOOP] for self-loop)

### 5. **DAG Statistics**
The tool can print detailed statistics including:
- Total number of nodes and edges
- Node type distribution
- Total WCET (Worst-Case Execution Time)
- Longest path length and nodes
- Source and sink nodes

## Usage

### Standalone Visualizer

```bash
# Basic usage - display interactive visualization
uv run python dag_visualizer.py input_dag.json

# Save to file
uv run python dag_visualizer.py input_dag.json -o dag.png

# With statistics
uv run python dag_visualizer.py input_dag.json --stats -o dag.png

# Different layouts
uv run python dag_visualizer.py input_dag.json -o dag_spring.png -l spring
uv run python dag_visualizer.py input_dag.json -o dag_circular.png -l circular

# Custom figure size and DPI
uv run python dag_visualizer.py input_dag.json -o dag_large.png --figsize 30,20 --dpi 300
```

### Integrated with Main Analyzer

```bash
# Visualize before analysis
uv run python main.py --visualize --viz-output dag.png

# Visualize with statistics
uv run python main.py --visualize --viz-stats --viz-output dag.png

# Visualize with critical path highlighted (automatic after analysis)
uv run python main.py --visualize --viz-output dag.png

# Use different layout
uv run python main.py --visualize --viz-layout spring --viz-output dag.png

# Combined with CPC analysis
uv run python main.py --verbose --visualize --viz-output dag.png

# Combined with time wall calculation
uv run python main.py --calc-time-wall --visualize --viz-output dag.png
```

## Command-Line Options

### Standalone Visualizer (`dag_visualizer.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `dag_path` | Path to DAG JSON file | Required |
| `-o, --output` | Output file path (e.g., dag.png) | None (interactive) |
| `-l, --layout` | Layout algorithm (hierarchical/spring/circular) | hierarchical |
| `--figsize` | Figure size as 'width,height' in inches | 20,12 |
| `--no-legend` | Don't show legend | False |
| `--dpi` | DPI for saved figure | 100 |
| `--stats` | Print DAG statistics | False |

### Main Analyzer (`main.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--visualize` | Generate DAG visualization | False |
| `--viz-output` | Path to save visualization | None (interactive) |
| `--viz-layout` | Layout algorithm | hierarchical |
| `--viz-stats` | Print DAG statistics | False |

## Examples

### Example 1: Quick Visualization

```bash
uv run python dag_visualizer.py input_dag.json -o output.png
```

This creates a basic hierarchical visualization saved as `output.png`.

### Example 2: High-Quality Publication Figure

```bash
uv run python dag_visualizer.py input_dag.json \
  -o high_res_dag.png \
  --figsize 30,20 \
  --dpi 300 \
  --stats
```

This creates a high-resolution (300 DPI) visualization with statistics printed to console.

### Example 3: Full Analysis with Visualization

```bash
uv run python main.py \
  --dag-path input_dag.json \
  --visualize \
  --viz-output analysis_dag.png \
  --viz-stats \
  --verbose \
  --num-cores 8
```

This runs a complete CPC analysis with verbose output, generates two visualizations:
1. `analysis_dag.png` - Initial DAG structure
2. `analysis_dag_critical_path.png` - DAG with critical path highlighted

### Example 4: Compare Different Layouts

```bash
# Hierarchical layout
uv run python dag_visualizer.py input_dag.json -o dag_hier.png -l hierarchical

# Spring layout
uv run python dag_visualizer.py input_dag.json -o dag_spring.png -l spring

# Circular layout
uv run python dag_visualizer.py input_dag.json -o dag_circular.png -l circular
```

## Understanding the Visualization

### Node Colors and Shapes
- **Sky Blue Nodes**: Regular computation tasks
- **Light Pink Nodes**: Provider nodes (group 1)
- **Light Salmon Nodes**: Provider nodes (group 2)
- **Hot Pink Nodes**: Provider nodes (group 3)
- **Deep Pink Nodes**: Provider nodes (group 4)
- **Gold Nodes**: Self-looping nodes with variable execution time

### Edges
- **Gray Arrows**: Regular dependencies
- **Red Thick Arrows**: Critical path (when highlighted)
- Arrow direction shows data flow from predecessor to successor

### Node Labels
Each node shows:
```
.../shortened/name
X.XX ms
[TYPE]
```

Where:
- First line: Node name (shortened for readability)
- Second line: Execution time in milliseconds
- Third line: Type indicator (optional)

### Legend Items
- **Default Node**: Regular computational tasks
- **Provider Node**: Tasks that provide data/services
- **Self-Loop Node**: Iterative algorithms with variable execution time
- **Critical Path**: (when highlighted) Longest path through the DAG

## Tips and Best Practices

### 1. **Choosing the Right Layout**
- **Hierarchical**: Best for understanding data flow and dependencies
- **Spring**: Best for identifying clusters and node relationships
- **Circular**: Best for compact representation and finding cycles

### 2. **Optimizing Visualization Quality**
For presentations or papers:
```bash
--figsize 30,20 --dpi 300
```

For quick analysis:
```bash
--figsize 20,12 --dpi 100
```

### 3. **Dealing with Large DAGs**
For DAGs with many nodes:
- Use larger figure sizes: `--figsize 40,30`
- Use hierarchical layout for best readability
- Consider splitting visualization by analyzing subsections

### 4. **Interactive Exploration**
Omit the `-o` option to display the visualization interactively:
```bash
uv run python dag_visualizer.py input_dag.json
```

This allows you to zoom, pan, and explore the graph before deciding on the final layout.

### 5. **Analyzing Critical Path**
Always run with `--visualize` through `main.py` to see the critical path:
```bash
uv run python main.py --visualize --viz-output dag.png
```

This generates two files:
- `dag.png`: Initial structure
- `dag_critical_path.png`: With critical path highlighted

## Programmatic Usage

You can also use the visualizer in your own Python scripts:

```python
from pathlib import Path
from dag_visualizer import DagVisualizer

# Create visualizer
viz = DagVisualizer(Path("input_dag.json"))

# Print statistics
viz.print_statistics()

# Generate visualization
viz.visualize(
    output_path=Path("output.png"),
    layout="hierarchical",
    figsize=(20, 12),
    show_legend=True,
    dpi=150
)

# Get statistics as dictionary
stats = viz.get_statistics()
print(f"Number of nodes: {stats['num_nodes']}")
print(f"Longest path length: {stats['longest_path_length']:.2f} ms")
```

### With Critical Path

```python
from pathlib import Path
from dag_visualizer import DagVisualizer
from cpc_analyzer import CpcGenericAnalyzer

# Load DAG (implement load_dag_from_file as in main.py)
dag, metadata, nodes_data = load_dag_from_file("input_dag.json")

# Analyze
analyzer = CpcGenericAnalyzer(dag, num_cores=8)
max_makespan, _ = analyzer.analyze()

# Visualize with critical path
viz = DagVisualizer(Path("input_dag.json"))
viz.visualize(
    output_path=Path("dag_critical.png"),
    layout="hierarchical",
    highlight_critical_path=True,
    critical_path=analyzer.critical_path,
    dpi=150
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'matplotlib'"
**Solution**: Install dependencies with `uv sync` and run with `uv run python ...`

### Issue: Visualization window doesn't appear (interactive mode)
**Solution**: 
- Make sure you're in an environment with display support
- Or use `-o output.png` to save to file instead

### Issue: Node labels are overlapping
**Solution**:
- Increase figure size: `--figsize 30,20`
- Try different layout: `-l spring`
- Use higher DPI for better resolution: `--dpi 300`

### Issue: Can't see critical path
**Solution**: 
- Make sure you're running through `main.py` with analysis
- Check that the second output file `*_critical_path.png` was generated

## Output Formats

The visualizer supports any format supported by matplotlib:
- PNG (recommended): `dag.png`
- PDF (vector, good for papers): `dag.pdf`
- SVG (vector, good for editing): `dag.svg`
- JPG: `dag.jpg`

Example:
```bash
uv run python dag_visualizer.py input_dag.json -o dag.pdf --dpi 300
```

## Performance Considerations

- For DAGs with < 100 nodes: Any layout works well
- For DAGs with 100-500 nodes: Use hierarchical layout, increase figsize
- For DAGs with > 500 nodes: Consider hierarchical layout with very large figsize or split into subgraphs

## Future Enhancements

Potential future features:
- Interactive web-based visualization
- Animation showing execution flow
- Real-time execution trace overlay
- Subgraph selection and filtering
- Provider group clustering visualization
- Execution time heatmap


