CAUSE_TO_ACTION = {
    "normal": {
        "action_id": "observe_only",
        "description": "No automated action. Continue observation.",
    },
    "bfd_outage": {
        "action_id": "restore_bfd_session",
        "description": "Re-enable or reconcile the BFD session on the implicated device.",
    },
    "blackhole": {
        "action_id": "rollback_blackhole_route",
        "description": "Roll back the offending FIB or policy change creating the blackhole.",
    },
    "ecmp_change": {
        "action_id": "restore_ecmp_hashing",
        "description": "Revert the ECMP loopback or hashing change and rebalance traffic.",
    },
    "interface_shutdown": {
        "action_id": "reroute_and_restore_interface",
        "description": "Reroute traffic away from the affected interface and restore interface state.",
    },
    "network_loop": {
        "action_id": "remove_looping_route",
        "description": "Remove the static route or configuration causing the forwarding loop.",
    },
}


def recommend_action(cause_name, target_device):
    template = CAUSE_TO_ACTION.get(cause_name, CAUSE_TO_ACTION["normal"]).copy()
    template["target_device"] = target_device
    return template


def validate_action(action, target_device, adjacency, node_names):
    if action["action_id"] == "observe_only":
        return False, "No-op recommendations are blocked for anomalous windows."

    degree_lookup = {
        node_name: int((adjacency[idx] > 0).sum() - 1)
        for idx, node_name in enumerate(node_names)
    }

    action_id = action["action_id"]
    if action_id == "restore_bfd_session":
        return target_device == "spine4-3464", "BFD recovery is only allowed on the spine4-3464 control-plane case."

    if action_id == "rollback_blackhole_route":
        return target_device == "leaf3", "Blackhole rollback is only valid on the hidden leaf3 route programming case."

    if action_id == "restore_ecmp_hashing":
        return target_device == "spine3", "ECMP restoration is only valid for the spine3 loopback change."

    if action_id == "reroute_and_restore_interface":
        valid = target_device in {"leaf4", "leaf7"} and degree_lookup.get(target_device, 0) >= 2
        return valid, "Interface remediation requires a leaf target with at least one alternate path."

    if action_id == "remove_looping_route":
        return target_device in {"dr02", "dr03"}, "Loop removal is restricted to the border-device pair."

    return False, "Unknown action."
