"""
DAG Visualization Module

This module provides visualization tools for Directed Acyclic Graphs (DAGs)
in the CPC analyzer, showing nodes, dependencies, execution times, and node types.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch


class DagVisualizer:
    """Visualize DAG structure with nodes, edges, and metadata."""
    
    # Color scheme for different node types
    NODE_COLORS = {
        "provider": "#FFB6C1",      # Light pink
        "default": "#87CEEB",        # Sky blue
        "self_loop": "#FFD700",      # Gold
    }
    
    # Provider group colors for additional distinction
    PROVIDER_GROUP_COLORS = [
        "#FFB6C1",  # Light pink
        "#FFA07A",  # Light salmon
        "#FF69B4",  # Hot pink
        "#FF1493",  # Deep pink
    ]
    
    def __init__(self, dag_path: Path):
        """Initialize the DAG visualizer.
        
        Args:
            dag_path: Path to the DAG JSON file
        """
        self.dag_path = Path(dag_path)
        self.graph = nx.DiGraph()
        self.nodes_data = []
        self.metadata = {}
        self._load_dag()
        
    def _parse_execution_time(self, raw_value) -> float:
        """Parse execution time from various formats."""
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_value)
            if match:
                return float(match.group(0))
        raise ValueError(f"Unsupported execution_time_ms value: {raw_value!r}")
    
    def _load_dag(self):
        """Load DAG from JSON file."""
        with self.dag_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        
        self.metadata = data.get("metadata", {})
        self.nodes_data = data.get("nodes", [])
        
        # Build the graph
        for node in self.nodes_data:
            node_id = node["id"]
            exec_time = self._parse_execution_time(node["execution_time_ms"])
            node_type = node.get("type", "default")
            provider_group = node.get("provider_group", None)
            
            self.graph.add_node(
                node_id,
                wcet=exec_time,
                type=node_type,
                provider_group=provider_group,
                label=self._create_node_label(node_id, exec_time, node_type)
            )
        
        # Add edges based on dependencies
        for node in self.nodes_data:
            node_id = node["id"]
            for dependency in node.get("dependencies", []):
                if dependency in self.graph:
                    self.graph.add_edge(dependency, node_id)
    
    def _create_node_label(self, node_id: str, exec_time: float, node_type: str) -> str:
        """Create a formatted label for the node."""
        # Shorten long node names
        name_parts = node_id.split("/")
        if len(name_parts) > 3:
            short_name = "/".join(["..."] + name_parts[-2:])
        else:
            short_name = node_id
        
        # Create label with execution time
        label = f"{short_name}\n{exec_time:.2f}ms"
        
        # Add type indicator
        if node_type == "self_loop":
            label += "\n[LOOP]"
        elif node_type == "provider":
            label += "\n[PROV]"
        
        return label
    
    def _get_node_color(self, node_id: str) -> str:
        """Get color for a node based on its type and provider group."""
        node_data = self.graph.nodes[node_id]
        node_type = node_data.get("type", "default")
        provider_group = node_data.get("provider_group")
        
        # Use provider group color if available
        if node_type == "provider" and provider_group is not None:
            group_idx = (provider_group - 1) % len(self.PROVIDER_GROUP_COLORS)
            return self.PROVIDER_GROUP_COLORS[group_idx]
        
        return self.NODE_COLORS.get(node_type, self.NODE_COLORS["default"])
    
    def _calculate_layout(self, layout: str = "hierarchical") -> Dict:
        """Calculate node positions for visualization.
        
        Args:
            layout: Layout algorithm to use ("hierarchical", "spring", "circular")
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if layout == "hierarchical":
            # Use topological sort for hierarchical layout
            return self._hierarchical_layout()
        elif layout == "spring":
            return nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == "circular":
            return nx.circular_layout(self.graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")
    
    def _hierarchical_layout(self) -> Dict:
        """Create a hierarchical layout based on topological levels."""
        # Calculate levels using longest path from sources
        levels = {}
        for node in nx.topological_sort(self.graph):
            predecessors = list(self.graph.predecessors(node))
            if not predecessors:
                levels[node] = 0
            else:
                levels[node] = max(levels[pred] for pred in predecessors) + 1
        
        # Group nodes by level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # Calculate positions
        pos = {}
        max_level = max(levels.values())
        
        for level, nodes in level_nodes.items():
            # Spread nodes horizontally within their level
            num_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = (i + 0.5) / num_nodes if num_nodes > 1 else 0.5
                y = 1.0 - (level / max_level) if max_level > 0 else 0.5
                pos[node] = (x, y)
        
        return pos
    
    def visualize(
        self,
        output_path: Optional[Path] = None,
        layout: str = "hierarchical",
        figsize: Tuple[int, int] = (20, 12),
        show_legend: bool = True,
        highlight_critical_path: bool = False,
        critical_path: Optional[List[str]] = None,
        dpi: int = 100,
    ):
        """Generate and display/save the DAG visualization.
        
        Args:
            output_path: Path to save the figure (if None, displays interactively)
            layout: Layout algorithm ("hierarchical", "spring", "circular")
            figsize: Figure size in inches (width, height)
            show_legend: Whether to show the legend
            highlight_critical_path: Whether to highlight the critical path
            critical_path: List of node IDs in the critical path
            dpi: Dots per inch for saved figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        pos = self._calculate_layout(layout)
        
        # Get node colors
        node_colors = [self._get_node_color(node) for node in self.graph.nodes()]
        
        # Draw edges
        edge_width = 1.5
        edge_color = "#888888"
        
        if highlight_critical_path and critical_path:
            # Draw normal edges first
            normal_edges = [
                (u, v) for u, v in self.graph.edges()
                if not (u in critical_path and v in critical_path and 
                       critical_path.index(v) == critical_path.index(u) + 1)
            ]
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=normal_edges,
                edge_color=edge_color, width=edge_width,
                arrows=True, arrowsize=15, ax=ax,
                connectionstyle="arc3,rad=0.1"
            )
            
            # Draw critical path edges in red
            critical_edges = [
                (critical_path[i], critical_path[i+1])
                for i in range(len(critical_path)-1)
                if critical_path[i] in self.graph and critical_path[i+1] in self.graph
                and self.graph.has_edge(critical_path[i], critical_path[i+1])
            ]
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=critical_edges,
                edge_color="red", width=edge_width * 2,
                arrows=True, arrowsize=20, ax=ax,
                connectionstyle="arc3,rad=0.1"
            )
        else:
            nx.draw_networkx_edges(
                self.graph, pos, edge_color=edge_color,
                width=edge_width, arrows=True, arrowsize=15, ax=ax,
                connectionstyle="arc3,rad=0.1"
            )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors,
            node_size=3000, ax=ax, alpha=0.9,
            edgecolors="black", linewidths=2
        )
        
        # Draw labels
        labels = nx.get_node_attributes(self.graph, "label")
        nx.draw_networkx_labels(
            self.graph, pos, labels, font_size=7,
            font_weight="bold", ax=ax
        )
        
        # Add title with metadata
        title = f"DAG Visualization: {self.dag_path.name}"
        if self.metadata:
            period = self.metadata.get("period_ms")
            deadline = self.metadata.get("deadline_ms")
            num_cores = self.metadata.get("num_cores")
            if period or deadline or num_cores:
                title += "\n"
                metadata_parts = []
                if period:
                    metadata_parts.append(f"Period: {period}ms")
                if deadline:
                    metadata_parts.append(f"Deadline: {deadline}ms")
                if num_cores:
                    metadata_parts.append(f"Cores: {num_cores}")
                title += " | ".join(metadata_parts)
        
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        
        # Add legend
        if show_legend:
            legend_elements = [
                plt.scatter([], [], c=self.NODE_COLORS["default"], s=200, 
                          edgecolors="black", linewidths=2, label="Default Node"),
                plt.scatter([], [], c=self.NODE_COLORS["provider"], s=200,
                          edgecolors="black", linewidths=2, label="Provider Node"),
                plt.scatter([], [], c=self.NODE_COLORS["self_loop"], s=200,
                          edgecolors="black", linewidths=2, label="Self-Loop Node"),
            ]
            
            if highlight_critical_path and critical_path:
                legend_elements.append(
                    plt.Line2D([0], [0], color="red", linewidth=3, label="Critical Path")
                )
            
            ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        
        # Add node count and edge count
        info_text = f"Nodes: {self.graph.number_of_nodes()} | Edges: {self.graph.number_of_edges()}"
        ax.text(0.99, 0.01, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.axis("off")
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"DAG visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_statistics(self) -> Dict:
        """Get statistics about the DAG.
        
        Returns:
            Dictionary containing DAG statistics
        """
        node_types = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "default")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        total_wcet = sum(
            self.graph.nodes[node].get("wcet", 0)
            for node in self.graph.nodes()
        )
        
        # Find source nodes (no predecessors)
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        # Find sink nodes (no successors)
        sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        # Calculate longest path
        try:
            longest_path = nx.dag_longest_path(self.graph, weight="wcet")
            longest_path_length = nx.dag_longest_path_length(self.graph, weight="wcet")
        except:
            longest_path = []
            longest_path_length = 0
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "total_wcet": total_wcet,
            "num_sources": len(sources),
            "num_sinks": len(sinks),
            "sources": sources,
            "sinks": sinks,
            "longest_path": longest_path,
            "longest_path_length": longest_path_length,
        }
    
    def print_statistics(self):
        """Print DAG statistics to console."""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("DAG STATISTICS")
        print("=" * 60)
        print(f"Total Nodes: {stats['num_nodes']}")
        print(f"Total Edges: {stats['num_edges']}")
        print(f"Source Nodes: {stats['num_sources']}")
        print(f"Sink Nodes: {stats['num_sinks']}")
        print(f"\nNode Types:")
        for node_type, count in stats['node_types'].items():
            print(f"  {node_type}: {count}")
        print(f"\nTotal WCET: {stats['total_wcet']:.2f} ms")
        print(f"Longest Path Length (WCET): {stats['longest_path_length']:.2f} ms")
        print(f"\nLongest Path:")
        for i, node in enumerate(stats['longest_path'], 1):
            wcet = self.graph.nodes[node].get("wcet", 0)
            print(f"  {i}. {node} ({wcet:.2f}ms)")
        print("=" * 60)


def main():
    """Command-line interface for DAG visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize DAG structure from JSON file"
    )
    parser.add_argument(
        "dag_path",
        type=Path,
        help="Path to DAG JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path for the visualization (e.g., dag.png)"
    )
    parser.add_argument(
        "-l", "--layout",
        choices=["hierarchical", "spring", "circular"],
        default="hierarchical",
        help="Layout algorithm to use (default: hierarchical)"
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="20,12",
        help="Figure size as 'width,height' in inches (default: 20,12)"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Don't show legend"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved figure (default: 100)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print DAG statistics"
    )
    
    args = parser.parse_args()
    
    # Parse figsize
    try:
        width, height = map(int, args.figsize.split(","))
        figsize = (width, height)
    except:
        print(f"Warning: Invalid figsize format '{args.figsize}', using default (20,12)")
        figsize = (20, 12)
    
    # Create visualizer
    visualizer = DagVisualizer(args.dag_path)
    
    # Print statistics if requested
    if args.stats:
        visualizer.print_statistics()
        print()
    
    # Generate visualization
    visualizer.visualize(
        output_path=args.output,
        layout=args.layout,
        figsize=figsize,
        show_legend=not args.no_legend,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()


