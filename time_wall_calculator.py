from __future__ import annotations
from math import floor

from cpc_analyzer import CpcGenericAnalyzer


class TimeWallCalculator:
    def __init__(self, normal_dag, safety_backup_dag, self_looping_node, deadline, m):
        self.normal_dag = normal_dag
        self.safety_backup_dag = safety_backup_dag  # Can be None
        self.vs = self_looping_node
        self.D = deadline
        self.m = m

    def _calculate_budget_for_dag(self, dag, mode_name: str) -> float:
        print(f"\n--- Calculating budget for {mode_name} DAG ---")

        try:
            temp_dag = {k: v.copy() for k, v in dag.items()}
            temp_dag[self.vs]['wcet'] = 0
            analyzer_for_emax = CpcGenericAnalyzer(temp_dag, self.m, verbose=False)
            path_with_vs = analyzer_for_emax.critical_path
            
            sum_wcet_on_path_except_vs = sum(temp_dag[n]['wcet'] for n in path_with_vs if n != self.vs)
            e_max = self.D - sum_wcet_on_path_except_vs
            if e_max <= 0:
                print(f"Warning: e_max ({e_max}) is non-positive. Task may be unschedulable.")
                return 0.0

        except Exception as e:
            print(f"Error calculating e_max: {e}")
            return 0.0
            
        print(f"Calculated e_max = {e_max:.2f}")

        try:
            dag_with_emax = {k: v.copy() for k, v in dag.items()}
            dag_with_emax[self.vs]['wcet'] = e_max
            analyzer_emax = CpcGenericAnalyzer(dag_with_emax, self.m, verbose=False)
            
            vs_segment_tuple = None
            for p_tuple in analyzer_emax.providers:
                if self.vs in p_tuple:
                    vs_segment_tuple = tuple(p_tuple)
                    break
            if not vs_segment_tuple:
                raise ValueError(f"Self-looping node {self.vs} not found in any provider segment.")

            vs_segment_index = analyzer_emax.providers.index(list(vs_segment_tuple))
            
            _, segment_responses_emax = analyzer_emax.analyze()
            sum_R_Vi_max_after = sum(
                segment_responses_emax[tuple(provider)]
                for idx, provider in enumerate(analyzer_emax.providers)
                if idx > vs_segment_index
            )
            
            analyzer_zero = CpcGenericAnalyzer(temp_dag, self.m, verbose=False)
            _, segment_responses_zero = analyzer_zero.analyze()
            sum_R_Vi_before = sum(
                segment_responses_zero[tuple(provider)]
                for idx, provider in enumerate(analyzer_zero.providers)
                if idx < vs_segment_index
            )

            R_Vs_init = self.D - sum_R_Vi_before - sum_R_Vi_max_after
            print(f"R(Vs)_init = {self.D} - {sum_R_Vi_before:.2f} - {sum_R_Vi_max_after:.2f} = {R_Vs_init:.2f}")
            if R_Vs_init <= 0:
                print("R(Vs)_init is non-positive. Task is likely unschedulable.")
                return 0.0

            provider_s = analyzer_emax.providers[vs_segment_index]
            lambda_s = tuple(provider_s)
            
            sum_ej_in_lambda_s_except_vs = sum(dag[n]['wcet'] for n in lambda_s if n != self.vs)
            consumer_s = analyzer_emax.consumers_F[lambda_s].union(analyzer_emax.consumers_G[lambda_s])
            sum_ej_in_consumer_s = sum(dag[n]['wcet'] for n in consumer_s)
            
            e_init = R_Vs_init - sum_ej_in_lambda_s_except_vs - (sum_ej_in_consumer_s / self.m)
            print(f"e_init = {R_Vs_init:.2f} - {sum_ej_in_lambda_s_except_vs:.2f} - ({sum_ej_in_consumer_s:.2f} / {self.m}) = {e_init:.2f}")

        except Exception as e:
            print(f"Could not calculate e_init due to an error: {e}. Defaulting to 0.")
            e_init = 0.0

        if e_init < 0:
            print("Warning: e_init is negative. Setting to 0.")
            e_init = 0.0

        e_s = self.normal_dag[self.vs]['wcet']
        low_bound_l = floor(e_init / e_s)
        high_bound_l = floor(e_max / e_s) + 1 # NOTE: +1 because we want to include the high bound in the search
        optimal_l = 0.0
        
        for _ in range(100):
            if high_bound_l < low_bound_l:
                break
            
            mid_l = floor((low_bound_l + high_bound_l) / 2)
            
            current_dag = {k: v.copy() for k, v in dag.items()}
            current_dag[self.vs]['wcet'] = mid_l * e_s

            try:
                analyzer = CpcGenericAnalyzer(current_dag, self.m, verbose=False)
                response_time, _ = analyzer.analyze()
                
                if response_time <= self.D:
                    optimal_l = mid_l
                    low_bound_l = mid_l
                else:
                    high_bound_l = mid_l - 1
            except Exception:
                high_bound_l = mid_l

        print(f"Found optimal budget for {mode_name}: {optimal_l * e_s:.2f}")
        return optimal_l * e_s

    def calculate_time_wall(self) -> float:
        e_norm = self._calculate_budget_for_dag(self.normal_dag, "Normal")
        
        if self.safety_backup_dag is None:
            print("\n--- Final Time Wall Calculation ---")
            print(f"e_norm (Normal mode budget): {e_norm:.2f}")
            print("Safety backup DAG not provided, using e_norm as time wall.")
            print(f"Time Wall: {e_norm:.2f}")
            return e_norm
        
        e_safe = self._calculate_budget_for_dag(self.safety_backup_dag, "Safety Backup")
        
        # The time wall must satisfy both normal and safety backup modes
        # Take the minimum to ensure both can meet the deadline
        time_wall = min(e_norm, e_safe)
        
        print("\n--- Final Time Wall Calculation ---")
        print(f"e_norm (Normal mode budget): {e_norm:.2f}")
        print(f"e_safe (Safety backup mode budget): {e_safe:.2f}")
        print(f"Time Wall (min of both): {time_wall:.2f}")
        
        return time_wall

