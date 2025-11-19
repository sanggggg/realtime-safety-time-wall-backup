import argparse
import json
import re
from pathlib import Path

from cpc_analyzer import CpcGenericAnalyzer
from time_wall_calculator import TimeWallCalculator


DEFAULT_INPUT_PATH = Path(__file__).with_name("input_dag.json")


def _parse_execution_time(raw_value):
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_value)
        if match:
            return float(match.group(0))
    raise ValueError(f"Unsupported execution_time_ms value: {raw_value!r}")


def load_dag_from_file(json_path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    nodes_data = data.get("nodes", [])
    dag = {}

    for node in nodes_data:
        node_id = node["id"]
        dag[node_id] = {
            "wcet": _parse_execution_time(node["execution_time_ms"]),
            "successors": [],
            "type": node.get("type", "default")
        }

    for node in nodes_data:
        for dependency in node.get("dependencies", []):
            if dependency not in dag:
                raise KeyError(f"Dependency {dependency} referenced by {node['id']} not found in DAG.")
            dag[dependency]["successors"].append(node["id"])

    return dag, data.get("metadata", {}), nodes_data


def find_self_looping_nodes(nodes_data):
    """Find all nodes with type 'self_loop'."""
    return [node["id"] for node in nodes_data if node.get("type") == "self_loop"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze DAG makespan using CPC Generic model.")
    parser.add_argument(
        "--dag-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to DAG description JSON (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        help="Override number of cores (otherwise taken from metadata.num_cores)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-provider CPC details",
    )
    parser.add_argument(
        "--calc-time-wall",
        action="store_true",
        help="Calculate time wall for self-looping nodes",
    )
    parser.add_argument(
        "--safety-dag-path",
        type=Path,
        help="Path to safety backup DAG JSON (optional; if not specified, only normal mode is analyzed)",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        help="Deadline in ms (otherwise taken from metadata.deadline_ms)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate DAG visualization",
    )
    parser.add_argument(
        "--viz-output",
        type=Path,
        help="Path to save visualization (e.g., dag.png). If not specified, displays interactively.",
    )
    parser.add_argument(
        "--viz-layout",
        choices=["hierarchical", "spring", "circular"],
        default="hierarchical",
        help="Layout algorithm for visualization (default: hierarchical)",
    )
    parser.add_argument(
        "--viz-stats",
        action="store_true",
        help="Print DAG statistics before visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dag, metadata, nodes_data = load_dag_from_file(args.dag_path)

    num_cores = args.num_cores if args.num_cores is not None else metadata.get("num_cores")
    if num_cores is None:
        raise ValueError(
            "Number of cores not provided. Set metadata.num_cores in the JSON or pass --num-cores."
        )
    num_cores = int(num_cores)

    print(f"Analyzing makespan with CPC model (Generic) on {num_cores} cores...")
    print(f"Loaded DAG from {args.dag_path}\n")

    analyzer = CpcGenericAnalyzer(dag, num_cores, verbose=args.verbose)

    print("--- CPC Model Constructed ---")
    # print(f"Critical Path: {analyzer.critical_path}")
    # for p in analyzer.providers:
    #     print(f"Provider {p}:")
    #     print(f"  F = {analyzer.consumers_F[tuple(p)]}")
    #     print(f"  G = {analyzer.consumers_G[tuple(p)]}")
    # print("-" * 20)

    max_makespan, _ = analyzer.analyze()

    print(f"\nFinal Calculated Max Makespan (CPC Generic): {max_makespan:.2f}")
    
    # Time wall calculation if requested
    if args.calc_time_wall:
        print("\n" + "=" * 60)
        print("TIME WALL CALCULATION")
        print("=" * 60)
        
        # Find self-looping nodes
        self_looping_nodes = find_self_looping_nodes(nodes_data)
        
        if not self_looping_nodes:
            print("No self-looping nodes found in the DAG.")
            return
        
        print(f"Found self-looping node(s): {self_looping_nodes}")
        
        # Get deadline
        deadline = args.deadline if args.deadline is not None else metadata.get("deadline_ms")
        if deadline is None:
            raise ValueError(
                "Deadline not provided. Set metadata.deadline_ms in the JSON or pass --deadline."
            )
        deadline = float(deadline)
        
        # Load safety backup DAG if provided
        if args.safety_dag_path:
            safety_dag, safety_metadata, _ = load_dag_from_file(args.safety_dag_path)
            print(f"Deadline: {deadline} ms")
            print(f"Safety backup DAG loaded from: {args.safety_dag_path}")
        else:
            safety_dag = None
            print(f"Deadline: {deadline} ms")
            print("No safety backup DAG specified, calculating time wall for normal mode only.")
        
        # Calculate time wall for each self-looping node
        for self_loop_node in self_looping_nodes:
            print(f"\n{'='*60}")
            print(f"Calculating time wall for: {self_loop_node}")
            print(f"{'='*60}")
            
            calculator = TimeWallCalculator(
                normal_dag=dag,
                safety_backup_dag=safety_dag,
                self_looping_node=self_loop_node,
                deadline=deadline,
                m=num_cores
            )
            
            time_wall = calculator.calculate_time_wall()
            
            print(f"\n{'*'*60}")
            print(f"FINAL TIME WALL for {self_loop_node}: {time_wall:.2f} ms")
            print(f"{'*'*60}")


if __name__ == "__main__":
    main()
 
