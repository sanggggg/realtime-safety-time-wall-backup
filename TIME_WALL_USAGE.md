# Time Wall Calculation Usage

## Overview

The `main.py` script now supports calculating time walls for self-looping nodes (nodes with `"type": "self_loop"`). This feature is useful for cyber-physical systems that need to determine safe execution time budgets for iterative algorithms.

## New Command-Line Arguments

### `--calc-time-wall`
Enable time wall calculation for self-looping nodes in the DAG.

### `--safety-dag-path SAFETY_DAG_PATH`
Path to the safety backup DAG JSON file. **This parameter is optional.** If not specified, only the normal mode time budget (e_norm) will be calculated and used as the time wall. When provided, both normal and safety backup modes are analyzed, and the time wall is set to the minimum of both budgets to ensure safety in both modes.

### `--deadline DEADLINE`
Deadline in milliseconds. If not specified, the deadline is read from `metadata.deadline_ms` in the input JSON file.

## Usage Examples

### Basic Time Wall Calculation (Normal Mode Only)

Calculate time wall using only the normal mode DAG and deadline from metadata:

```bash
python3 main.py --calc-time-wall
```

This will calculate only `e_norm` and use it as the time wall.

### With Custom Deadline

Override the deadline from the command line:

```bash
python3 main.py --calc-time-wall --deadline 150.0
```

### With Separate Safety Backup DAG

Use a different DAG file for safety backup mode analysis:

```bash
python3 main.py --calc-time-wall --safety-dag-path safety_backup_dag.json
```

### Full Example with All Options

```bash
python3 main.py \
  --dag-path input_dag.json \
  --safety-dag-path safety_dag.json \
  --num-cores 8 \
  --deadline 133.0 \
  --calc-time-wall \
  --verbose
```

## Self-Looping Nodes

Self-looping nodes are identified by the `"type": "self_loop"` field in the node definition. For example:

```json
{
  "id": "/localization/pose_estimator/ndt_scan_matcher",
  "name": "/localization/pose_estimator/ndt_scan_matcher",
  "execution_time_ms": "2.95/iter",
  "type": "self_loop",
  "provider_group": 1,
  "dependencies": [
    "/localization/util/random_downsample_filter",
    "/sensing/lidar/concatenate_data"
  ]
}
```

## Output

When `--calc-time-wall` is enabled, the program will:

1. Run the standard CPC analysis
2. Identify all self-looping nodes in the DAG
3. For each self-looping node:
   - Calculate the optimal time budget for normal mode (e_norm)
   - If `--safety-dag-path` is provided:
     - Calculate the optimal time budget for safety backup mode (e_safe)
     - Compute the final time wall as the minimum of both budgets
   - If `--safety-dag-path` is NOT provided:
     - Use e_norm as the final time wall
4. Display the final time wall result in milliseconds

## Example Output

### Normal Mode Only (No Safety Backup DAG)

```
============================================================
TIME WALL CALCULATION
============================================================
Found self-looping node(s): ['/localization/pose_estimator/ndt_scan_matcher']
Deadline: 133.0 ms
No safety backup DAG specified, calculating time wall for normal mode only.

============================================================
Calculating time wall for: /localization/pose_estimator/ndt_scan_matcher
============================================================

--- Calculating budget for Normal DAG ---
Calculated e_max = 99.65
Found optimal budget for Normal: 99.65

--- Final Time Wall Calculation ---
e_norm (Normal mode budget): 99.65
Safety backup DAG not provided, using e_norm as time wall.
Time Wall: 99.65

************************************************************
FINAL TIME WALL for /localization/pose_estimator/ndt_scan_matcher: 99.65 ms
************************************************************
```

### With Safety Backup DAG

```
============================================================
TIME WALL CALCULATION
============================================================
Found self-looping node(s): ['/localization/pose_estimator/ndt_scan_matcher']
Deadline: 133.0 ms
Safety backup DAG loaded from: input_dag.json

============================================================
Calculating time wall for: /localization/pose_estimator/ndt_scan_matcher
============================================================

--- Calculating budget for Normal DAG ---
Calculated e_max = 99.65
Found optimal budget for Normal: 99.65

--- Calculating budget for Safety Backup DAG ---
Calculated e_max = 99.65
Found optimal budget for Safety Backup: 99.65

--- Final Time Wall Calculation ---
e_norm (Normal mode budget): 99.65
e_safe (Safety backup mode budget): 99.65
Time Wall (min of both): 99.65

************************************************************
FINAL TIME WALL for /localization/pose_estimator/ndt_scan_matcher: 99.65 ms
************************************************************
```

## Implementation Details

The time wall calculation uses the `TimeWallCalculator` class from `time_wall_calculator.py`, which:

1. Uses binary search to find the optimal execution time budget
2. If only normal mode is analyzed (no safety backup DAG):
   - Returns e_norm as the time wall
3. If both modes are analyzed (safety backup DAG provided):
   - Ensures the system can meet its deadline in both normal and safety backup modes
   - Takes the minimum of both budgets as the final time wall to guarantee safety in both modes

## Notes

- The deadline must be specified either via `--deadline` or in the JSON metadata
- If no self-looping nodes are found, the program will notify you and skip time wall calculation
- The time wall calculation is performed after the standard CPC analysis
- Multiple self-looping nodes are supported; the time wall is calculated for each one independently

