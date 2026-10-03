from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpatialQueryStats:
    nodes_visited: int = 0
    internal_nodes_visited: int = 0
    leaf_nodes_visited: int = 0
    mbr_tests: int = 0
    bound_evaluations: int = 0
    nodes_pruned: int = 0
    entries_examined: int = 0
    distance_evaluations: int = 0
    polygon_tests: int = 0
    heap_pushes: int = 0
    heap_pops: int = 0
    frontier_peak: int = 0
    candidates_peak: int = 0
    result_count: int = 0

    def reset(self) -> None:
        self.nodes_visited = 0
        self.internal_nodes_visited = 0
        self.leaf_nodes_visited = 0
        self.mbr_tests = 0
        self.bound_evaluations = 0
        self.nodes_pruned = 0
        self.entries_examined = 0
        self.distance_evaluations = 0
        self.polygon_tests = 0
        self.heap_pushes = 0
        self.heap_pops = 0
        self.frontier_peak = 0
        self.candidates_peak = 0
        self.result_count = 0

    def visit_node(self, *, is_leaf: bool) -> None:
        self.nodes_visited += 1

        if is_leaf:
            self.leaf_nodes_visited += 1
        else:
            self.internal_nodes_visited += 1

    def observe_frontier(self, size: int) -> None:
        self.frontier_peak = max(self.frontier_peak, size)

    def observe_candidates(self, size: int) -> None:
        self.candidates_peak = max(self.candidates_peak, size)
