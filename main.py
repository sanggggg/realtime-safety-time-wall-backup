import argparse
import collections
import json
import re
from pathlib import Path
from functools import lru_cache


def _parse_execution_time(raw_value):
    """
    Normalize execution_time_ms field which may be a float or a string like '2.95/iter'.
    Returns the first float found in the string.
    """
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_value)
        if match:
            return float(match.group(0))
    raise ValueError(f"Unsupported execution_time_ms value: {raw_value!r}")


DEFAULT_INPUT_PATH = Path(__file__).with_name("input_dag.json")


def load_dag_from_file(json_path):
    """
    Parse the CPC input DAG JSON into the internal representation used by the analyzer.
    Returns (dag_dict, metadata_dict)
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    nodes_data = data.get("nodes", [])
    dag = {}

    # Initialize nodes
    for node in nodes_data:
        node_id = node["id"]
        dag[node_id] = {
            "wcet": _parse_execution_time(node["execution_time_ms"]),
            "successors": []
        }

    # Populate successors using the dependencies list (which lists predecessors)
    for node in nodes_data:
        for dependency in node.get("dependencies", []):
            if dependency not in dag:
                raise KeyError(f"Dependency {dependency} referenced by {node['id']} not found in DAG.")
            dag[dependency]["successors"].append(node["id"])

    return dag, data.get("metadata", {})


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
    return parser.parse_args()

# --- Helper Functions ---
def get_predecessors(dag, node):
    preds = []
    for n, details in dag.items():
        if node in details['successors']:
            preds.append(n)
    return preds

def get_ancestors(dag, node):
    ancestors = set()
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

# --- Core Implementation ---
class CpcGenericAnalyzer:
    def __init__(self, dag, m):
        self.dag = dag
        self.m = m
        self.nodes = list(dag.keys())
        self.predecessors = {n: get_predecessors(dag, n) for n in self.nodes}
        self.all_ancestors = {n: get_ancestors(dag, n) for n in self.nodes}
        
        # 1. 임계 경로 찾기 및 CPC 모델 구성
        self.critical_path = self._find_critical_path()
        self.providers, self.consumers_F, self.consumers_G = self._construct_cpc_model()

        # 3. f(v) 계산
        self.finish_times = self._calculate_all_finish_times()

    def _find_critical_path(self):
        # 위상 정렬 및 가장 긴 경로 찾기
        in_degree = {n: 0 for n in self.nodes}
        for n in self.nodes:
            for succ in self.dag[n]['successors']:
                in_degree[succ] += 1
        
        q = collections.deque([n for n in self.nodes if in_degree[n] == 0])
        
        dist = {n: details['wcet'] for n, details in self.dag.items()}
        path_pred = {n: None for n in self.nodes}

        while q:
            u = q.popleft()
            for v in self.dag[u]['successors']:
                if dist[u] + self.dag[v]['wcet'] > dist[v]:
                    dist[v] = dist[u] + self.dag[v]['wcet']
                    path_pred[v] = u
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        
        # sink 노드 찾기 (후속 노드가 없는 노드)
        sink_node = next(n for n in self.nodes if not self.dag[n]['successors'])
        
        # 경로 역추적
        cp = []
        curr = sink_node
        while curr:
            cp.append(curr)
            curr = path_pred[curr]
        return cp[::-1]

    def _construct_cpc_model(self):
        # 논문의 Algorithm 1 구현
        providers = []
        cp_nodes = set(self.critical_path)
        non_critical_nodes = set(self.nodes) - cp_nodes
        
        i = 0
        while i < len(self.critical_path):
            provider = [self.critical_path[i]]
            i = i + 1
            # 연속된 임계 경로 노드들을 하나의 프로바이더로 묶음
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
                # F(θi): θi+1의 조상이 되는 비-임계 노드
                f_theta_i = get_ancestors(self.dag, theta_i_plus_1_head).intersection(remaining_V_minus)
            else:
                f_theta_i = set()
            
            consumers_F[tuple(theta_i)] = f_theta_i
            
            # G(θi): F(θi)와 병렬 실행 가능하지만 θi+1을 직접 지연시키지 않는 노드
            g_theta_i = set()
            for vj in f_theta_i:
                for vk in remaining_V_minus:
                     # C(vj) \ V-
                    if vk not in get_ancestors(self.dag, vj) and vk not in get_descendants_and_self(self.dag, vj):
                        g_theta_i.add(vk)
            
            # WARN: 왜인지 모르겠는데, 논문에서는 v_j 와 병렬가능한 애들의 합집합으로 나오는 바람에 v_j들 (F(θi))이 같이 들어가 버린다.
            # 이건 아닌거 같아서 빼준다, 논문에서 오타낸듯?
            g_theta_i = g_theta_i - f_theta_i
            consumers_G[tuple(theta_i)] = g_theta_i
            
            remaining_V_minus -= f_theta_i

        return providers, consumers_F, consumers_G


    def _calculate_all_finish_times(self):
        """
        논문의 Equation (3), (4)를 반영하여 모든 노드의 finish time f(v)를 계산합니다.
        재귀 호출과 메모이제이션(lru_cache)을 사용합니다.
        """
        # lru_cache를 사용하여 finish_time_recursive의 결과를 캐싱합니다.
        # 이렇게 하면 각 노드의 finish_time이 한 번만 계산됩니다.
        for node in self.nodes:
            self._finish_time_recursive(node)
        
        # 캐시된 결과를 딕셔너리로 변환하여 반환
        return {node: self._finish_time_recursive(node) for node in self.nodes}

    @lru_cache(maxsize=None)
    def _finish_time_recursive(self, v):
        """ Equation (3)을 재귀적으로 계산하는 함수 """
        wcet_v = self.dag[v]['wcet']
        
        # max_{vk in pre(vj)} {f(vk)}
        max_pred_finish = 0
        if self.predecessors[v]:
            max_pred_finish = max(self._finish_time_recursive(p) for p in self.predecessors[v])
            
        # 간섭(Interference) 항 계산
        interference_delay = 0
        
        # 비-임계 경로 노드에 대해서만 간섭을 계산
        if v not in self.critical_path:
            # Equation (4)에 따라 간섭 집합 I(v) 계산
            interference_set_I_v = self._get_interference_set(v)
            
            # 간섭 작업량 계산
            interference_workload = sum(self.dag[i_node]['wcet'] for i_node in interference_set_I_v)
            
            # 간섭으로 인한 지연 시간 계산 (m-1개 코어에서 처리)
            # 제네릭 케이스이므로 1/(m-1)로 나눔
            if self.m > 1 and interference_workload > 0:
                interference_delay = interference_workload / (self.m - 1)

        # f(vj) = Cj + max_{...} + interference
        return wcet_v + max_pred_finish + interference_delay

    def _get_interference_set(self, v):
        """
        Equation (4)를 구현: 노드 v에 대한 간섭 집합 I(v)를 반환합니다.
        I(vj) = {vk | vk not in X* AND vk not in U_{vr in anc(vj)} C(vr), vk in C(vj)}
        """
        # C(v): v와 동시 실행 가능한 노드 집합
        # C(vj) = {vk | vk != (anc(vj) U des(vj) U {vj}))}
        v_ancestors = self.all_ancestors[v]
        v_descendants = get_descendants_and_self(self.dag, v)
        concurrent_to_v = set(self.nodes) - v_ancestors - v_descendants

        # U_{vr in anc(vj)} C(vr): v의 조상들과 동시 실행 가능한 노드 집합
        # 간섭 집합 I(vr)은 논문에서 I(vj)와 동일한 방식으로 계산됨
        # 제네릭 분석에서는 이 부분을 단순화하여 v의 조상을 방해할 수 없는 노드, 
        # 즉 v와 진정한 의미에서 동시 실행되는 노드만 고려
        # I(vj)는 v와 동시 실행 가능(concurrent_to_v)하면서
        # v의 조상들과는 의존성이 없는 비-임계 노드들.
        interference_set = set()
        for vk in concurrent_to_v:
            if vk in self.critical_path:
                continue

            # vk가 v의 어떤 조상과도 의존성이 없는지 확인
            # (vk가 어떤 조상의 자손도 아니고, 어떤 조상의 조상도 아님)
            is_independent_of_ancestors = True
            for ancestor in v_ancestors:
                if vk in self.all_ancestors[ancestor] or ancestor in self.all_ancestors[vk]:
                    is_independent_of_ancestors = False
                    break
            
            if is_independent_of_ancestors:
                 interference_set.add(vk)
        
        return interference_set

    def analyze(self):
        total_response_time = 0
        
        for i in range(len(self.providers)):
            theta_i = tuple(self.providers[i])
            f_theta_i = self.consumers_F[theta_i]
            g_theta_i = self.consumers_G[theta_i]
            
            # Li, Wi 계산
            L_i = sum(self.dag[n]['wcet'] for n in theta_i)
            W_i = L_i + sum(self.dag[n]['wcet'] for n in f_theta_i) + sum(self.dag[n]['wcet'] for n in g_theta_i)

            # f(θi) 계산
            finish_time_provider = self.finish_times[theta_i[-1]]

            # 4. αi, βi 계산
            alpha_i, beta_i = self._calculate_alpha_beta(theta_i, f_theta_i, g_theta_i, finish_time_provider)
            
            # 5. 지연 시간 계산 (Equation 2)
            delay_term = 0
            interfering_workload = W_i - L_i - alpha_i - beta_i
            if interfering_workload > 0 and self.m > 0:
                delay_term = interfering_workload / self.m
            
            response_time_i = L_i + delay_term + beta_i
            total_response_time += response_time_i
            
            # Provider별 계산 결과 출력
            print(f"--- Provider θ_{i+1}: {list(theta_i)} ---")
            print(f"  L_{i}: {L_i}, W_{i}: {W_i}")
            print(f"  F(θ_{i}): {list(f_theta_i)}")
            print(f"  G(θ_{i}): {list(g_theta_i)}")
            print(f"  f(θ_{i}): {finish_time_provider}")
            print(f"  α_{i}: {alpha_i:.2f}, β_{i}: {beta_i:.2f}")
            print(f"  Delay term for this provider: {delay_term:.2f} + {beta_i:.2f}")
            print(f"  Response time for this segment: {response_time_i:.2f}")
            
        return total_response_time

    def _calculate_alpha_beta(self, theta_i, f_theta_i, g_theta_i, f_provider):
        # Equation (7) for alpha_i
        alpha_i = 0
        consumers = f_theta_i.union(g_theta_i)
        for v in consumers:
            f_v = self.finish_times[v]
            wcet_v = self.dag[v]['wcet']
            if f_v <= f_provider:
                alpha_i += wcet_v  # Case Va
            elif f_v > f_provider and (f_v - wcet_v) < f_provider:
                alpha_i += f_provider - (f_v - wcet_v) # Case Vb
        
        # Equation (8), (9) for beta_i
        beta_i = 0
        interfering_consumers = {v for v in f_theta_i if self.finish_times[v] > f_provider}
        if not interfering_consumers:
            return alpha_i, 0

        # 가장 긴 간섭 경로 찾기 (여기서는 가장 늦게 끝나는 노드를 기준으로 단순화)
        # 실제 구현은 Equation 8의 재귀적 경로 탐색이 필요
        latest_interfering_consumer = max(interfering_consumers, key=lambda v: self.finish_times[v])
        
        # 간섭 경로 역추적 (단순화된 버전)
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

        # Equation (9)
        for v in beta_path:
            f_v = self.finish_times[v]
            wcet_v = self.dag[v]['wcet']
            start_v = f_v - wcet_v
            if start_v >= f_provider:
                beta_i += wcet_v
            else:
                beta_i += (f_v - f_provider)
                
        return alpha_i, beta_i

# Helper for G(θi) calculation (not in class for simplicity)
def get_descendants_and_self(dag, node):
    descendants = {node}
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
    
def main():
    args = parse_args()
    dag, metadata = load_dag_from_file(args.dag_path)

    num_cores = args.num_cores if args.num_cores is not None else metadata.get("num_cores")
    if num_cores is None:
        raise ValueError(
            "Number of cores not provided. Set metadata.num_cores in the JSON or pass --num-cores."
        )
    num_cores = int(num_cores)

    print(f"Analyzing makespan with CPC model (Generic) on {num_cores} cores...")
    print(f"Loaded DAG from {args.dag_path}\n")

    analyzer = CpcGenericAnalyzer(dag, num_cores)

    print("--- CPC Model Constructed ---")
    print(f"Critical Path: {analyzer.critical_path}")
    for p in analyzer.providers:
        print(f"Provider {p}:")
        print(f"  F = {analyzer.consumers_F[tuple(p)]}")
        print(f"  G = {analyzer.consumers_G[tuple(p)]}")
    print("-" * 20)

    max_makespan = analyzer.analyze()

    print(f"\nFinal Calculated Max Makespan (CPC Generic): {max_makespan:.2f}")


if __name__ == "__main__":
    main()