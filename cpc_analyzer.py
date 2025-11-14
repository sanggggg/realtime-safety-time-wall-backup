import collections
from functools import lru_cache
from typing import Dict, Iterable, List, Set, Tuple


Dag = Dict[str, Dict[str, Iterable[str]]]


def get_predecessors(dag: Dag, node: str) -> List[str]:
    preds: List[str] = []
    for n, details in dag.items():
        if node in details['successors']:
            preds.append(n)
    return preds


def get_ancestors(dag: Dag, node: str) -> Set[str]:
    ancestors: Set[str] = set()
    q = collections.deque(get_predecessors(dag, node))
    visited = set(q)
    while q:
        curr = q.popleft()
        ancestors.add(curr)
        for pred in get_predecessors(dag, curr):
            if pred not in visited:
                visited.add(pred)
                q.append(pred)
    return ancestors


def get_descendants_and_self(dag: Dag, node: str) -> Set[str]:
    descendants: Set[str] = {node}
    q = collections.deque(dag[node]['successors'])
    visited = set(q)
    while q:
        curr = q.popleft()
        descendants.add(curr)
        for succ in dag[curr]['successors']:
            if succ not in visited:
                visited.add(succ)
                q.append(succ)
    return descendants


class CpcGenericAnalyzer:
    def __init__(self, dag: Dag, m: int, *, verbose: bool = False):
        self.dag = dag
        self.m = m
        self.verbose = verbose
        self.nodes = list(dag.keys())
        self.predecessors = {n: get_predecessors(dag, n) for n in self.nodes}
        self.all_ancestors = {n: get_ancestors(dag, n) for n in self.nodes}
        
        self.critical_path = self._find_critical_path()
        self.providers, self.consumers_F, self.consumers_G = self._construct_cpc_model()

        self.finish_times = self._calculate_all_finish_times()

    def _find_critical_path(self) -> List[str]:
        # Find self_loop node(s)
        self_loop_nodes = [n for n in self.nodes if self.dag[n].get('type') == 'self_loop']
        
        # Calculate longest path from source to all nodes (forward pass)
        in_degree = {n: 0 for n in self.nodes}
        for n in self.nodes:
            for succ in self.dag[n]['successors']:
                in_degree[succ] += 1
        
        source_nodes = [n for n in self.nodes if in_degree[n] == 0]
        sink_nodes = [n for n in self.nodes if not self.dag[n]['successors']]
        
        # Forward pass: longest distance from source to each node
        q = collections.deque(source_nodes)
        dist_forward = {n: 0 for n in self.nodes}
        for n in source_nodes:
            dist_forward[n] = self.dag[n]['wcet']
        
        path_pred = {n: None for n in self.nodes}
        in_degree_copy = in_degree.copy()

        while q:
            u = q.popleft()
            for v in self.dag[u]['successors']:
                if dist_forward[u] + self.dag[v]['wcet'] > dist_forward[v]:
                    dist_forward[v] = dist_forward[u] + self.dag[v]['wcet']
                    path_pred[v] = u
                in_degree_copy[v] -= 1
                if in_degree_copy[v] == 0:
                    q.append(v)
        
        # Backward pass: longest distance from each node to sink
        # Build reverse graph
        reverse_graph = {n: [] for n in self.nodes}
        out_degree = {n: 0 for n in self.nodes}
        for n in self.nodes:
            for succ in self.dag[n]['successors']:
                reverse_graph[succ].append(n)
                out_degree[n] += 1
        
        q_back = collections.deque(sink_nodes)
        dist_backward = {n: 0 for n in self.nodes}
        for n in sink_nodes:
            dist_backward[n] = self.dag[n]['wcet']
        
        path_succ = {n: None for n in self.nodes}
        out_degree_copy = out_degree.copy()
        
        while q_back:
            v = q_back.popleft()
            for u in reverse_graph[v]:
                if dist_backward[v] + self.dag[u]['wcet'] > dist_backward[u]:
                    dist_backward[u] = dist_backward[v] + self.dag[u]['wcet']
                    path_succ[u] = v
                out_degree_copy[u] -= 1
                if out_degree_copy[u] == 0:
                    q_back.append(u)
        
        # If there's a self_loop node, force critical path through it
        if self_loop_nodes:
            # Find the self_loop node with maximum path length through it
            best_self_loop = None
            best_length = -1
            
            for sl_node in self_loop_nodes:
                # Total length through this self_loop node
                # dist_forward includes wcet of sl_node, dist_backward includes wcet of sl_node
                # So we subtract it once to avoid double counting
                total_length = dist_forward[sl_node] + dist_backward[sl_node] - self.dag[sl_node]['wcet']
                
                if total_length > best_length:
                    best_length = total_length
                    best_self_loop = sl_node
            
            # Reconstruct path: source -> best_self_loop -> sink
            # Path from source to best_self_loop (backwards)
            path_to_sl = []
            curr = best_self_loop
            while curr:
                path_to_sl.append(curr)
                curr = path_pred[curr]
            path_to_sl = path_to_sl[::-1]
            
            # Path from best_self_loop to sink (forwards)
            path_from_sl = []
            curr = path_succ[best_self_loop]
            while curr:
                path_from_sl.append(curr)
                curr = path_succ[curr]
            
            return path_to_sl + path_from_sl
        
        # If no self_loop node, use original algorithm
        sink_node = sink_nodes[0] if sink_nodes else self.nodes[-1]
        
        cp = []
        curr = sink_node
        while curr:
            cp.append(curr)
            curr = path_pred[curr]
        return cp[::-1]

    def _construct_cpc_model(self):
        providers: List[List[str]] = []
        cp_nodes = set(self.critical_path)
        non_critical_nodes = set(self.nodes) - cp_nodes
        
        i = 0
        while i < len(self.critical_path):
            provider = [self.critical_path[i]]
            i = i + 1
            while i < len(self.critical_path) and set(self.predecessors[self.critical_path[i]]) == {self.critical_path[i-1]}:
                provider.append(self.critical_path[i])
                i += 1
            providers.append(provider)
            
        consumers_F = {}
        consumers_G = {}
        
        remaining_V_minus = set(non_critical_nodes)
        
        for i in range(len(providers)):
            theta_i = providers[i]
            
            if i + 1 < len(providers):
                theta_i_plus_1_head = providers[i+1][0]
                f_theta_i = get_ancestors(self.dag, theta_i_plus_1_head).intersection(remaining_V_minus)
            else:
                f_theta_i = set()
            
            consumers_F[tuple(theta_i)] = f_theta_i
            
            g_theta_i = set()
            for vj in f_theta_i:
                for vk in remaining_V_minus:
                    if vk not in get_ancestors(self.dag, vj) and vk not in get_descendants_and_self(self.dag, vj):
                        g_theta_i.add(vk)
            
            g_theta_i = g_theta_i - f_theta_i
            consumers_G[tuple(theta_i)] = g_theta_i
            
            remaining_V_minus -= f_theta_i

        return providers, consumers_F, consumers_G

    def _calculate_all_finish_times(self):
        for node in self.nodes:
            self._finish_time_recursive(node)
        
        return {node: self._finish_time_recursive(node) for node in self.nodes}

    @lru_cache(maxsize=None)
    def _finish_time_recursive(self, v: str) -> float:
        wcet_v = self.dag[v]['wcet']
        
        max_pred_finish = 0.0
        if self.predecessors[v]:
            max_pred_finish = max(self._finish_time_recursive(p) for p in self.predecessors[v])
            
        interference_delay = 0.0
        
        if v not in self.critical_path:
            interference_set_I_v = self._get_interference_set(v)
            interference_workload = sum(self.dag[i_node]['wcet'] for i_node in interference_set_I_v)
            
            if self.m > 1 and interference_workload > 0:
                interference_delay = interference_workload / (self.m - 1)

        return wcet_v + max_pred_finish + interference_delay

    def _get_interference_set(self, v: str) -> Set[str]:
        v_ancestors = self.all_ancestors[v]
        v_descendants = get_descendants_and_self(self.dag, v)
        concurrent_to_v = set(self.nodes) - v_ancestors - v_descendants

        interference_set = set()
        for vk in concurrent_to_v:
            if vk in self.critical_path:
                continue

            is_independent_of_ancestors = True
            for ancestor in v_ancestors:
                if vk in self.all_ancestors[ancestor] or ancestor in self.all_ancestors[vk]:
                    is_independent_of_ancestors = False
                    break
            
            if is_independent_of_ancestors:
                 interference_set.add(vk)
        
        return interference_set

    def _calculate_alpha_beta(self, theta_i, f_theta_i, g_theta_i, f_provider):
        alpha_i = 0.0
        consumers = f_theta_i.union(g_theta_i)
        for v in consumers:
            f_v = self.finish_times[v]
            wcet_v = self.dag[v]['wcet']
            if f_v <= f_provider:
                alpha_i += wcet_v
            elif f_v > f_provider and (f_v - wcet_v) < f_provider:
                alpha_i += f_provider - (f_v - wcet_v)
        
        beta_i = 0.0
        interfering_consumers = {v for v in f_theta_i if self.finish_times[v] > f_provider}
        if not interfering_consumers:
            return alpha_i, 0.0

        latest_interfering_consumer = max(interfering_consumers, key=lambda v: self.finish_times[v])
        
        beta_path = [latest_interfering_consumer]
        curr = latest_interfering_consumer
        while self.predecessors[curr]:
            pred = max(self.predecessors[curr], key=lambda p: self.finish_times.get(p, 0))
            if pred in interfering_consumers:
                beta_path.append(pred)
                curr = pred
            else:
                break
        beta_path = beta_path[::-1]

        for v in beta_path:
            f_v = self.finish_times[v]
            wcet_v = self.dag[v]['wcet']
            start_v = f_v - wcet_v
            if start_v >= f_provider:
                beta_i += wcet_v
            else:
                beta_i += (f_v - f_provider)
                
        return alpha_i, beta_i

    def analyze(self):
        total_response_time = 0.0
        segment_response_times: Dict[Tuple[str, ...], float] = {}

        for i in range(len(self.providers)):
            theta_i = tuple(self.providers[i])
            f_theta_i = self.consumers_F[theta_i]
            g_theta_i = self.consumers_G[theta_i]
            
            L_i = sum(self.dag[n]['wcet'] for n in theta_i)
            W_i = L_i + sum(self.dag[n]['wcet'] for n in f_theta_i) + sum(self.dag[n]['wcet'] for n in g_theta_i)

            finish_time_provider = self.finish_times[theta_i[-1]]

            alpha_i, beta_i = self._calculate_alpha_beta(theta_i, f_theta_i, g_theta_i, finish_time_provider)
            
            delay_term = 0.0
            interfering_workload = W_i - L_i - alpha_i - beta_i
            if interfering_workload > 0 and self.m > 0:
                delay_term = interfering_workload / self.m
            
            response_time_i = L_i + delay_term + beta_i
            segment_response_times[theta_i] = response_time_i
            total_response_time += response_time_i
            
            if self.verbose:
                print(f"--- Provider θ_{i+1}: {list(theta_i)} ---")
                print(f"  L_{i}: {L_i}, W_{i}: {W_i}")
                print(f"  F(θ_{i}): {list(f_theta_i)}")
                print(f"  G(θ_{i}): {list(g_theta_i)}")
                print(f"  f(θ_{i}): {finish_time_provider}")
                print(f"  α_{i}: {alpha_i:.2f}, β_{i}: {beta_i:.2f}")
                print(f"  Delay term for this provider: {delay_term:.2f} + {beta_i:.2f}")
                print(f"  Response time for this segment: {response_time_i:.2f}")
            
        return total_response_time, segment_response_times

