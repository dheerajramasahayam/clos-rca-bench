def get_remediation(root_cause):
    """
    Returns automated remediation actions based on identified root cause.
    """
    remediation_map = {
        'none': "No action needed.",
        'congestion': "Trigger traffic engineering: reroute traffic to alternative paths via BGP local preference adjustments.",
        'misconfig': "Roll back last committed configuration on the affected router interface.",
        'hardware_failure': "Isolate the faulty node and schedule hardware replacement. Divert all traffic to redundant links.",
        'bgp_instability': "Reset BGP session and apply route dampening to stabilize prefix flapping."
    }
    return remediation_map.get(root_cause, "Unknown root cause. Manual inspection required.")

if __name__ == "__main__":
    causes = ['congestion', 'misconfig', 'hardware_failure', 'bgp_instability', 'none']
    for cause in causes:
        print(f"Root Cause: {cause:15} | Remediation: {get_remediation(cause)}")
