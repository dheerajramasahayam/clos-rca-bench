from dataclasses import dataclass, replace
import json
from pathlib import Path

import networkx as nx
import numpy as np


RAW_ROOT = Path("dataset/cisco_topology_benchmark/raw")
CANONICAL_SCENARIO = "200121_0803_ecmp"
SERVICE_NODES = ["dr02", "dr03", "leaf3", "leaf4", "leaf5", "leaf7", "leaf8"]


@dataclass(frozen=True)
class TwinState:
    failed_edges: frozenset
    blocked_destinations: frozenset
    preferred_spine: str | None
    disabled_nodes: frozenset


def _sorted_edge(a, b):
    return tuple(sorted((a, b)))


def load_canonical_topology(node_names):
    data = json.loads((RAW_ROOT / CANONICAL_SCENARIO / "cdp_map.json").read_text())
    graph = nx.Graph()
    graph.add_nodes_from(node_names)
    interface_map = {}

    for source, ports in data.get("devices", {}).items():
        if source not in node_names:
            continue
        for interface_name, details in ports.items():
            target = details.get("target_device")
            if target not in node_names:
                continue
            graph.add_edge(source, target)
            interface_map[(source, interface_name)] = target

    return graph, interface_map


def base_state():
    return TwinState(
        failed_edges=frozenset(),
        blocked_destinations=frozenset(),
        preferred_spine=None,
        disabled_nodes=frozenset(),
    )


def demand_pairs(graph):
    active_nodes = [node for node in SERVICE_NODES if node in graph.nodes]
    return [(src, dst) for src in active_nodes for dst in active_nodes if src != dst]


def _active_graph(graph, state):
    active = graph.copy()
    active.remove_nodes_from([node for node in state.disabled_nodes if node in active])
    active.remove_edges_from([edge for edge in state.failed_edges if active.has_edge(*edge)])
    return active


def _edge_capacities(graph):
    capacities = {}
    for src, dst in graph.edges():
        if src.startswith("spine") or dst.startswith("spine"):
            capacities[_sorted_edge(src, dst)] = 6.0
        elif src.startswith("dr") or dst.startswith("dr"):
            capacities[_sorted_edge(src, dst)] = 4.0
        else:
            capacities[_sorted_edge(src, dst)] = 2.0
    return capacities


def _candidate_paths(active_graph, src, dst, preferred_spine):
    try:
        paths = list(nx.all_shortest_paths(active_graph, src, dst))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    if preferred_spine:
        biased = [path for path in paths if preferred_spine in path[1:-1]]
        if biased:
            return biased
    return paths


def _edge_disjoint_count(active_graph, src, dst):
    try:
        return min(4, sum(1 for _ in nx.edge_disjoint_paths(active_graph, src, dst)))
    except (nx.NetworkXException, nx.NodeNotFound):
        return 0


def score_state(graph, state):
    active_graph = _active_graph(graph, state)
    demands = demand_pairs(graph)
    capacities = _edge_capacities(graph)

    reachable = 0
    total_diversity = 0.0
    failed_demands = 0
    edge_loads = {edge: 0.0 for edge in capacities}

    for src, dst in demands:
        if src not in active_graph or dst not in active_graph or dst in state.blocked_destinations:
            failed_demands += 1
            continue

        paths = _candidate_paths(active_graph, src, dst, state.preferred_spine)
        if not paths:
            failed_demands += 1
            continue

        reachable += 1
        total_diversity += _edge_disjoint_count(active_graph, src, dst)
        per_path = 1.0 / len(paths)
        for path in paths:
            for edge_idx in range(len(path) - 1):
                edge = _sorted_edge(path[edge_idx], path[edge_idx + 1])
                if edge in edge_loads:
                    edge_loads[edge] += per_path

    total_demands = len(demands)
    overload_sum = 0.0
    overloaded_edges = 0
    for edge, load in edge_loads.items():
        excess = max(0.0, load - capacities[edge])
        overload_sum += excess
        overloaded_edges += int(excess > 0.0)

    return {
        "Reachability": reachable / total_demands if total_demands else 0.0,
        "AvgPathDiversity": total_diversity / reachable if reachable else 0.0,
        "FailedDemands": failed_demands,
        "OverloadedEdges": overloaded_edges,
        "OverloadExcess": overload_sum,
        "BlastRadius": failed_demands / total_demands if total_demands else 0.0,
    }


def inject_fault(graph, interface_map, cause_name, target_device, target_interface):
    state = base_state()

    if cause_name == "interface_shutdown":
        neighbor = interface_map.get((target_device, target_interface))
        if neighbor and graph.has_edge(target_device, neighbor):
            return replace(state, failed_edges=frozenset({_sorted_edge(target_device, neighbor)}))
        return state

    if cause_name == "blackhole":
        return replace(state, blocked_destinations=frozenset({target_device}))

    if cause_name == "bfd_outage":
        return replace(state, disabled_nodes=frozenset({target_device}))

    if cause_name == "ecmp_change":
        return replace(state, preferred_spine=target_device)

    return state


def apply_action(graph, state, action_id, target_device):
    if action_id == "restore_bfd_session" and target_device == "spine4-3464":
        return replace(state, disabled_nodes=frozenset())

    if action_id == "rollback_blackhole_route" and target_device == "leaf3":
        blocked = set(state.blocked_destinations)
        blocked.discard("leaf3")
        return replace(state, blocked_destinations=frozenset(blocked))

    if action_id == "restore_ecmp_hashing" and target_device == "spine3":
        return replace(state, preferred_spine=None)

    if action_id == "reroute_and_restore_interface" and target_device in {"leaf4", "leaf7"}:
        return replace(state, failed_edges=frozenset())

    return state


def evaluate_recovery(graph, interface_map, cause_name, ground_target, target_interface, action_id, predicted_target, safe):
    nominal = score_state(graph, base_state())
    fault_state = inject_fault(graph, interface_map, cause_name, ground_target, target_interface)
    fault_metrics = score_state(graph, fault_state)

    if safe:
        recovered_state = apply_action(graph, fault_state, action_id, predicted_target)
    else:
        recovered_state = fault_state
    recovered_metrics = score_state(graph, recovered_state)

    reachability_gain = recovered_metrics["Reachability"] - fault_metrics["Reachability"]
    blast_reduction = fault_metrics["BlastRadius"] - recovered_metrics["BlastRadius"]
    overload_reduction = fault_metrics["OverloadExcess"] - recovered_metrics["OverloadExcess"]

    success = (
        recovered_metrics["Reachability"] >= nominal["Reachability"] - 1e-9
        and recovered_metrics["BlastRadius"] <= fault_metrics["BlastRadius"]
        and recovered_metrics["OverloadExcess"] <= fault_metrics["OverloadExcess"] + 1e-9
    )

    return {
        "NominalReachability": nominal["Reachability"],
        "FaultReachability": fault_metrics["Reachability"],
        "RecoveredReachability": recovered_metrics["Reachability"],
        "FaultBlastRadius": fault_metrics["BlastRadius"],
        "RecoveredBlastRadius": recovered_metrics["BlastRadius"],
        "FaultPathDiversity": fault_metrics["AvgPathDiversity"],
        "RecoveredPathDiversity": recovered_metrics["AvgPathDiversity"],
        "FaultOverloadExcess": fault_metrics["OverloadExcess"],
        "RecoveredOverloadExcess": recovered_metrics["OverloadExcess"],
        "ReachabilityGain": reachability_gain,
        "BlastRadiusReduction": blast_reduction,
        "OverloadReduction": overload_reduction,
        "RecoverySuccess": int(success),
    }
