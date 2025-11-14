# CPC DAG Analyzer with Time Wall Calculation

A tool for analyzing Directed Acyclic Graphs (DAGs) using the CPC (Critical Path and Consumers) model, with support for calculating time walls for self-looping nodes in cyber-physical systems.

## Features

- **CPC Generic Model Analysis**: Analyzes DAG makespan using the CPC model for multicore systems
- **Time Wall Calculation**: Calculates safe execution time budgets for self-looping nodes
- **Flexible Input**: Supports JSON-based DAG descriptions
- **Configurable**: Command-line options for cores, deadline, and analysis modes
- **Safety Analysis**: Supports separate normal and safety backup DAG configurations

## Installation

This project uses Python 3 and requires the following dependencies (managed via `pyproject.toml`):

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Quick Start

### Basic CPC Analysis

Analyze a DAG's makespan:

```bash
python3 main.py
```

### Time Wall Calculation

Calculate time wall for self-looping nodes:

```bash
python3 main.py --calc-time-wall
```

## Usage

```bash
python3 main.py [OPTIONS]
```

### Options

- `--dag-path DAG_PATH`: Path to DAG description JSON (default: `input_dag.json`)
- `--num-cores NUM_CORES`: Override number of cores (otherwise from metadata)
- `--verbose`: Print per-provider CPC details
- `--calc-time-wall`: Calculate time wall for self-looping nodes
- `--safety-dag-path SAFETY_DAG_PATH`: Path to safety backup DAG JSON (optional; if not specified, only normal mode is analyzed)
- `--deadline DEADLINE`: Deadline in ms (otherwise from metadata)

### Examples

**Basic analysis:**
```bash
python3 main.py --dag-path input_dag.json --num-cores 8
```

**Time wall (normal mode only):**
```bash
python3 main.py --calc-time-wall
```

**Time wall with custom deadline:**
```bash
python3 main.py --calc-time-wall --deadline 150.0
```

**Time wall with safety backup DAG:**
```bash
python3 main.py --calc-time-wall --safety-dag-path safety_backup_dag.json
```

**Full analysis with verbose output:**
```bash
python3 main.py --calc-time-wall --verbose --num-cores 8
```

## Input Format

The input JSON file should contain:

```json
{
  "metadata": {
    "period_ms": 133,
    "deadline_ms": 133,
    "num_cores": 8
  },
  "nodes": [
    {
      "id": "node_id",
      "name": "node_name",
      "execution_time_ms": 4.033,
      "type": "provider",
      "dependencies": []
    }
  ]
}
```

### Node Types

- `default`: Regular node
- `provider`: Provider node in the CPC model
- `self_loop`: Self-looping node (e.g., iterative algorithms)

## Project Structure

```
.
├── main.py                      # Main entry point
├── cpc_analyzer.py              # CPC model analysis implementation
├── time_wall_calculator.py      # Time wall calculation for self-looping nodes
├── input_dag.json               # Example input DAG
├── README.md                    # This file
└── pyproject.toml               # Project dependencies
```

## Components

### CPC Analyzer (`cpc_analyzer.py`)

Implements the CPC Generic model for analyzing DAG makespan on multicore systems:
- Critical path identification
- Provider/consumer segmentation
- Response time calculation with interference analysis

### Time Wall Calculator (`time_wall_calculator.py`)

Calculates safe execution time budgets for self-looping nodes:
- Binary search optimization
- Support for normal mode (required) and optional safety backup mode
- When safety backup DAG is provided, returns minimum of both budgets
- When safety backup DAG is not provided, returns normal mode budget only
- Deadline-aware budget allocation

## Self-Looping Nodes

Self-looping nodes are special nodes that perform iterative computations. They are identified by `"type": "self_loop"` in the node definition. The time wall calculator determines the maximum safe execution time budget for these nodes while ensuring the system meets its deadline.

Example:
```json
{
  "id": "/localization/pose_estimator/ndt_scan_matcher",
  "execution_time_ms": "2.95/iter",
  "type": "self_loop",
  "dependencies": [...]
}
```

## Output

The tool provides:

1. **CPC Analysis Results**:
   - Critical path
   - Provider segments
   - Consumer sets (F and G)
   - Maximum makespan

2. **Time Wall Results** (when `--calc-time-wall` is enabled):
   - Identified self-looping nodes
   - Optimal time budget for normal mode (e_norm)
   - If safety backup DAG provided:
     - Optimal time budget for safety backup mode (e_safe)
     - Final time wall (minimum of both)
   - If safety backup DAG not provided:
     - Final time wall (e_norm only)

## References

Based on research in cyber-physical systems and real-time scheduling analysis.

## License

See project license file.

