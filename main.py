import collections
from functools import lru_cache

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
                for vk in self.nodes:
                     # C(vj) \ V-
                    if vk not in get_ancestors(self.dag, vj) and vk not in get_descendants_and_self(self.dag, vj):
                       if vk in remaining_V_minus:
                           g_theta_i.add(vk)
            
            g_theta_i = g_theta_i - f_theta_i
            consumers_G[tuple(theta_i)] = g_theta_i
            
            remaining_V_minus -= f_theta_i

        return providers, consumers_F, consumers_G

    @lru_cache(maxsize=None)
    def _calculate_finish_time(self, v):
        # Equation (3) - 제네릭 케이스 (간단한 버전)
        # 실제 논문은 Ac(vj)\λ* < m-1 조건을 확인하지만, 제네릭 분석을 위해
        # 모든 non-critical 노드가 간섭을 겪는다고 가정
        if v in self.critical_path:
             # 임계 경로 노드는 이전 임계 경로 노드가 끝나면 바로 시작 (간섭 없음)
             preds = self.predecessors[v]
             max_pred_finish = 0
             if preds:
                 max_pred_finish = max(self._calculate_finish_time(p) for p in preds)
             return max_pred_finish + self.dag[v]['wcet']

        # 비-임계 경로 노드
        concurrent_nodes = self.nodes # 단순화를 위해 모든 노드가 잠재적 간섭자로 가정
        interference_workload = sum(self.dag[c]['wcet'] for c in concurrent_nodes if c not in self.critical_path and c != v)
        interference_delay = interference_workload / (self.m - 1) if self.m > 1 else float('inf')
        
        max_pred_finish = 0
        if self.predecessors[v]:
            max_pred_finish = max(self._calculate_finish_time(p) for p in self.predecessors[v])
            
        return max_pred_finish + self.dag[v]['wcet'] + interference_delay

    def _calculate_all_finish_times(self):
        # f(v) 계산은 위상 정렬 순서로 진행하면 재귀 없이 효율적으로 가능
        # 여기서는 재귀적 구현을 단순화하여 보여줌
        # 실제로는 매우 비관적인 가정을 사용한 계산이 됨
        # 제네릭 α, β 계산을 위한 단순화된 f(v)
        f = {}
        # 소스 노드부터 시작
        q = collections.deque([n for n in self.nodes if not self.predecessors[n]])
        visited_topo = set(q)
        f.update({n: self.dag[n]['wcet'] for n in q})
        
        while q:
            u = q.popleft()
            for v in self.dag[u]['successors']:
                if v not in visited_topo:
                    max_pred_f = max(f.get(p, 0) for p in self.predecessors[v])
                    f[v] = max_pred_f + self.dag[v]['wcet']
                    visited_topo.add(v)
                    q.append(v)
        return f

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
    
# --- 예제 실행 ---
if __name__ == "__main__":
    dag_example = {
        'v1': {'wcet': 1, 'successors': ['v2', 'v3', 'v5', 'v6']},
        'v2': {'wcet': 7, 'successors': ['v4']},
        'v3': {'wcet': 2, 'successors': ['v4']},
        'v4': {'wcet': 3, 'successors': ['v8']},
        'v5': {'wcet': 2, 'successors': ['v7']},
        'v6': {'wcet': 4, 'successors': ['v7']},
        'v7': {'wcet': 3, 'successors': ['v8']},
        'v8': {'wcet': 4, 'successors': []}
    }
    num_cores = 2

    print(f"Analyzing makespan with CPC model (Generic) on {num_cores} cores...\n")
    analyzer = CpcGenericAnalyzer(dag_example, num_cores)
    
    print("--- CPC Model Constructed ---")
    print(f"Critical Path: {analyzer.critical_path}")
    for p in analyzer.providers:
        print(f"Provider {p}:")
        print(f"  F = {analyzer.consumers_F[tuple(p)]}")
        print(f"  G = {analyzer.consumers_G[tuple(p)]}")
    print("-" * 20)

    max_makespan = analyzer.analyze()

    print(f"\nFinal Calculated Max Makespan (CPC Generic): {max_makespan:.2f}")