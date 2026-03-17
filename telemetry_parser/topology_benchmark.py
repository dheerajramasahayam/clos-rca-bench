import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_ROOT = Path("dataset/cisco_topology_benchmark/raw")
OUTPUT_ROOT = Path("dataset/cisco_topology_benchmark/processed")

BIN_SECONDS = 10
WINDOW_SIZE = 6
WINDOW_STEP = 3
RANDOM_STATE = 42

SCENARIOS = [
    "200121_0803_ecmp",
    "S-200202_1940_bfd-1",
    "S-200202_2014_evtmix-1",
    "S-200202_2155_evtmix-1",
    "S-200205_0138_evtmix-1",
    "S-200206_1309_blackhole-1",
    "S-200206_1852_evtmix-1",
    "S-200206_1929_evtmix-1",
]

CAUSE_ORDER = [
    "normal",
    "bfd_outage",
    "blackhole",
    "ecmp_change",
    "interface_shutdown",
]

TARGET_DEVICE_ORDER = [
    "none",
    "leaf3",
    "leaf4",
    "leaf7",
    "spine3",
    "spine4-3464",
]

DATA_RATE_FILE = (
    "Cisco-IOS-XR-infra-statsd-oper_"
    "infra-statistics_interfaces_interface_latest_data-rate.csv"
)
GENERIC_COUNTERS_FILE = (
    "Cisco-IOS-XR-infra-statsd-oper_"
    "infra-statistics_interfaces_interface_latest_generic-counters.csv"
)
BFD_FILE = "Cisco-IOS-XR-ip-bfd-oper_bfd_session-briefs_session-brief.csv"
BGP_FILE = (
    "Cisco-IOS-XR-ipv4-bgp-oper_"
    "bgp_instances_instance_instance-active_default-vrf_process-info.csv"
)
FIB_FILE = "Cisco-IOS-XR-fib-common-oper_fib-statistics_nodes_node_drops.csv"
INTERFACE_BRIEF_FILE = "Cisco-IOS-XR-pfi-im-cmd-oper_interfaces_interface-briefs_interface-brief.csv"
CPU_FILE = "Cisco-IOS-XR-wdsysmon-fd-oper_system-monitoring_cpu-utilization.csv"


def _rename_time_column(frame):
    first = frame.columns[0]
    frame = frame.rename(columns={first: "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame["bin_start"] = (frame["timestamp"].astype("int64") // 10**9 // BIN_SECONDS) * BIN_SECONDS
    return frame


def _safe_numeric(frame, columns):
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def _delta_by_key(frame, key_column, value_columns):
    frame = frame.sort_values(["timestamp", key_column]).copy()
    for column in value_columns:
        frame[column] = frame.groupby(key_column)[column].diff().clip(lower=0).fillna(0.0)
    return frame


def _read_zip_csv(zip_path, inner_name):
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(inner_name)]
        if not members:
            raise FileNotFoundError(f"{inner_name} not found inside {zip_path}")
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle, low_memory=False)


def _empty_feature_frame(columns):
    return pd.DataFrame(columns=["bin_start", *columns])


def aggregate_data_rate(frame):
    frame = _rename_time_column(frame)
    numeric_columns = [
        "input-data-rate",
        "output-data-rate",
        "input-packet-rate",
        "output-packet-rate",
        "input-load",
        "output-load",
    ]
    frame = _safe_numeric(frame, numeric_columns)
    return (
        frame.groupby("bin_start")
        .agg(
            input_data_rate_sum=("input-data-rate", "sum"),
            output_data_rate_sum=("output-data-rate", "sum"),
            input_packet_rate_sum=("input-packet-rate", "sum"),
            output_packet_rate_sum=("output-packet-rate", "sum"),
            input_load_max=("input-load", "max"),
            output_load_max=("output-load", "max"),
        )
        .reset_index()
    )


def aggregate_generic_counters(frame):
    frame = _rename_time_column(frame)
    key_column = "interface-name"
    numeric_columns = [
        "carrier-transitions",
        "crc-errors",
        "input-drops",
        "input-errors",
        "output-drops",
        "output-errors",
    ]
    frame = _safe_numeric(frame, numeric_columns)
    frame = _delta_by_key(frame, key_column, numeric_columns)
    return (
        frame.groupby("bin_start")
        .agg(
            carrier_transitions_delta=("carrier-transitions", "sum"),
            crc_errors_delta=("crc-errors", "sum"),
            input_drops_delta=("input-drops", "sum"),
            input_errors_delta=("input-errors", "sum"),
            output_drops_delta=("output-drops", "sum"),
            output_errors_delta=("output-errors", "sum"),
        )
        .reset_index()
    )


def aggregate_bfd(frame):
    frame = _rename_time_column(frame)
    frame["bfd_up"] = (frame.get("state", "") == "bfd-mgmt-session-state-up").astype(int)
    frame["bfd_down"] = 1 - frame["bfd_up"]
    return (
        frame.groupby("bin_start")
        .agg(bfd_up_count=("bfd_up", "sum"), bfd_down_count=("bfd_down", "sum"))
        .reset_index()
    )


def aggregate_bgp(frame):
    frame = _rename_time_column(frame)
    numeric_columns = [
        "global/established-neighbors-count-total",
        "global/neighbors-count-total",
        "global/nexthop-count",
        "vrf/path-count",
        "vrf/update-messages-received",
    ]
    frame = _safe_numeric(frame, numeric_columns)
    return (
        frame.groupby("bin_start")
        .agg(
            bgp_established_neighbors=("global/established-neighbors-count-total", "max"),
            bgp_neighbors_total=("global/neighbors-count-total", "max"),
            bgp_nexthop_count=("global/nexthop-count", "max"),
            bgp_path_count=("vrf/path-count", "sum"),
            bgp_updates_received=("vrf/update-messages-received", "sum"),
        )
        .reset_index()
    )


def aggregate_fib(frame):
    frame = _rename_time_column(frame)
    key_column = "node-name"
    numeric_columns = [
        "discard-packets",
        "incomplete-adjacency-packets",
        "total-number-of-drop-packets",
        "unresolved-prefix-packets",
    ]
    frame = _safe_numeric(frame, numeric_columns)
    frame = _delta_by_key(frame, key_column, numeric_columns)
    return (
        frame.groupby("bin_start")
        .agg(
            fib_discard_packets_delta=("discard-packets", "sum"),
            fib_incomplete_adjacency_delta=("incomplete-adjacency-packets", "sum"),
            fib_total_drop_packets_delta=("total-number-of-drop-packets", "sum"),
            fib_unresolved_prefix_delta=("unresolved-prefix-packets", "sum"),
        )
        .reset_index()
    )


def aggregate_interface_brief(frame):
    frame = _rename_time_column(frame)
    frame["is_up"] = (frame.get("state", "") == "im-state-up").astype(int)
    frame["is_admin_down"] = (frame.get("actual-state", "") == "im-state-admin-down").astype(int)
    frame["is_oper_down"] = 1 - frame["is_up"]
    return (
        frame.groupby("bin_start")
        .agg(
            interface_up_count=("is_up", "sum"),
            interface_oper_down_count=("is_oper_down", "sum"),
            interface_admin_down_count=("is_admin_down", "sum"),
        )
        .reset_index()
    )


def aggregate_cpu(frame):
    frame = _rename_time_column(frame)
    numeric_columns = ["total-cpu-five-minute", "total-cpu-one-minute"]
    frame = _safe_numeric(frame, numeric_columns)
    return (
        frame.groupby("bin_start")
        .agg(
            cpu_total_five_mean=("total-cpu-five-minute", "mean"),
            cpu_total_one_mean=("total-cpu-one-minute", "mean"),
            cpu_total_one_max=("total-cpu-one-minute", "max"),
        )
        .reset_index()
    )


def build_device_frame(zip_path):
    feature_specs = [
        (
            DATA_RATE_FILE,
            aggregate_data_rate,
            [
                "input_data_rate_sum",
                "output_data_rate_sum",
                "input_packet_rate_sum",
                "output_packet_rate_sum",
                "input_load_max",
                "output_load_max",
            ],
        ),
        (
            GENERIC_COUNTERS_FILE,
            aggregate_generic_counters,
            [
                "carrier_transitions_delta",
                "crc_errors_delta",
                "input_drops_delta",
                "input_errors_delta",
                "output_drops_delta",
                "output_errors_delta",
            ],
        ),
        (
            BFD_FILE,
            aggregate_bfd,
            ["bfd_up_count", "bfd_down_count"],
        ),
        (
            BGP_FILE,
            aggregate_bgp,
            [
                "bgp_established_neighbors",
                "bgp_neighbors_total",
                "bgp_nexthop_count",
                "bgp_path_count",
                "bgp_updates_received",
            ],
        ),
        (
            FIB_FILE,
            aggregate_fib,
            [
                "fib_discard_packets_delta",
                "fib_incomplete_adjacency_delta",
                "fib_total_drop_packets_delta",
                "fib_unresolved_prefix_delta",
            ],
        ),
        (
            INTERFACE_BRIEF_FILE,
            aggregate_interface_brief,
            [
                "interface_up_count",
                "interface_oper_down_count",
                "interface_admin_down_count",
            ],
        ),
        (
            CPU_FILE,
            aggregate_cpu,
            ["cpu_total_five_mean", "cpu_total_one_mean", "cpu_total_one_max"],
        ),
    ]

    feature_frames = []
    for inner_name, aggregator, output_columns in feature_specs:
        try:
            raw_frame = _read_zip_csv(zip_path, inner_name)
            feature_frames.append(aggregator(raw_frame))
        except FileNotFoundError:
            feature_frames.append(_empty_feature_frame(output_columns))

    merged = feature_frames[0]
    for frame in feature_frames[1:]:
        merged = merged.merge(frame, on="bin_start", how="outer")

    merged = merged.sort_values("bin_start").infer_objects(copy=False).fillna(0.0)
    return merged


def list_scenario_devices(scenario_dir):
    yang_dir = scenario_dir / "yang_models"
    return sorted(path.stem for path in yang_dir.glob("*.zip"))


def infer_labels(events):
    events = events.sort_values("timestamp").reset_index(drop=True)
    traffic_start = float(events.loc[events["event"] == "ixchariot_traffic", "timestamp"].iloc[0])
    traffic_stop = float(events.loc[events["event"] == "ixchariot_traffic_stopped", "timestamp"].iloc[0])

    if (events["event"] == "add_blackhole").any():
        row = events.loc[events["event"] == "add_blackhole"].iloc[0]
        return {
            "cause": "blackhole",
            "target_device": row["device"],
            "target_interface": row.get("interface", ""),
            "anomaly_start": float(row["timestamp"]),
            "anomaly_end": traffic_stop,
        }

    if (events["event"] == "set_loopback").any():
        row = events.loc[events["event"] == "set_loopback"].iloc[0]
        return {
            "cause": "ecmp_change",
            "target_device": row["device"],
            "target_interface": row.get("interface", ""),
            "anomaly_start": float(row["timestamp"]),
            "anomaly_end": traffic_stop,
        }

    if (events["event"] == "shutdown_interface").any():
        down_row = events.loc[events["event"] == "shutdown_interface"].iloc[0]
        enable_rows = events.loc[events["event"] == "enable_interface"]
        anomaly_end = float(enable_rows.iloc[0]["timestamp"]) if not enable_rows.empty else traffic_stop
        return {
            "cause": "interface_shutdown",
            "target_device": down_row["device"],
            "target_interface": down_row.get("interface", ""),
            "anomaly_start": float(down_row["timestamp"]),
            "anomaly_end": anomaly_end,
        }

    if (events["event"] == "add_network_loop").any():
        add_row = events.loc[events["event"] == "add_network_loop"].iloc[0]
        remove_rows = events.loc[
            (events["event"] == "remove_network_loop") & (events["timestamp"] > add_row["timestamp"])
        ]
        anomaly_end = float(remove_rows.iloc[0]["timestamp"]) if not remove_rows.empty else traffic_stop
        return {
            "cause": "network_loop",
            "target_device": "dr02",
            "target_interface": "",
            "anomaly_start": float(add_row["timestamp"]),
            "anomaly_end": anomaly_end,
        }

    if (events["event"] == "enable_bfd").any():
        row = events.loc[events["event"] == "enable_bfd"].iloc[0]
        return {
            "cause": "bfd_outage",
            "target_device": row["device"],
            "target_interface": row.get("interface", ""),
            "anomaly_start": traffic_start,
            "anomaly_end": float(row["timestamp"]),
        }

    raise ValueError(f"Unable to infer labels from events: {events['event'].tolist()}")


def build_node_list():
    observed = set()
    hidden_targets = set()

    for scenario_name in SCENARIOS:
        scenario_dir = RAW_ROOT / scenario_name
        observed.update(list_scenario_devices(scenario_dir))
        events = pd.read_csv(scenario_dir / "events.csv")
        labels = infer_labels(events)
        hidden_targets.add(labels["target_device"])

    nodes = sorted(observed.union(hidden_targets))
    return nodes


def build_adjacency(node_names):
    index = {name: idx for idx, name in enumerate(node_names)}
    adjacency = np.eye(len(node_names), dtype=np.float32)

    for scenario_name in SCENARIOS:
        cdp_path = RAW_ROOT / scenario_name / "cdp_map.json"
        if not cdp_path.exists():
            continue
        with open(cdp_path, "r", encoding="utf-8") as handle:
            cdp_map = json.load(handle)

        for source, ports in cdp_map.get("devices", {}).items():
            if source not in index:
                continue
            for details in ports.values():
                target = details.get("target_device")
                if target not in index:
                    continue
                src_idx = index[source]
                dst_idx = index[target]
                adjacency[src_idx, dst_idx] = 1.0
                adjacency[dst_idx, src_idx] = 1.0

    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.diag(np.power(degree, -0.5, where=degree > 0))
    normalized = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    return normalized.astype(np.float32)


def build_scenario_tensor(scenario_name, node_names, feature_names):
    scenario_dir = RAW_ROOT / scenario_name
    events = pd.read_csv(scenario_dir / "events.csv")
    labels = infer_labels(events)

    traffic_event = events.loc[events["event"] == "ixchariot_traffic"].iloc[0]
    traffic_stop = events.loc[events["event"] == "ixchariot_traffic_stopped"].iloc[0]

    start_bin = int(float(traffic_event["timestamp"]) // BIN_SECONDS) * BIN_SECONDS
    stop_bin = int(float(traffic_stop["timestamp"]) // BIN_SECONDS) * BIN_SECONDS
    all_bins = np.arange(start_bin, stop_bin + BIN_SECONDS, BIN_SECONDS)

    device_frames = {}
    for device_name in list_scenario_devices(scenario_dir):
        frame = build_device_frame(scenario_dir / "yang_models" / f"{device_name}.zip")
        frame = frame.set_index("bin_start").reindex(all_bins).fillna(0.0).reset_index()
        frame["observation_mask"] = 1.0
        device_frames[device_name] = frame

    window_records = []
    scenario_tensor = []

    cause_id = CAUSE_ORDER.index(labels["cause"])
    target_id = TARGET_DEVICE_ORDER.index(labels["target_device"])

    for start_idx in range(0, len(all_bins) - WINDOW_SIZE + 1, WINDOW_STEP):
        end_idx = start_idx + WINDOW_SIZE
        window_start = int(all_bins[start_idx])
        window_end = int(all_bins[end_idx - 1])

        node_tensor = np.zeros((WINDOW_SIZE, len(node_names), len(feature_names)), dtype=np.float32)
        for node_idx, node_name in enumerate(node_names):
            if node_name in device_frames:
                node_frame = device_frames[node_name].iloc[start_idx:end_idx]
                values = node_frame[feature_names].to_numpy(dtype=np.float32)
            else:
                values = np.zeros((WINDOW_SIZE, len(feature_names)), dtype=np.float32)
            node_tensor[:, node_idx, :] = values

        is_anomaly = int(window_end >= labels["anomaly_start"] and window_start < labels["anomaly_end"])
        scenario_tensor.append(node_tensor)
        window_records.append(
            {
                "scenario": scenario_name,
                "window_start": window_start,
                "window_end": window_end,
                "anomaly_label": is_anomaly,
                "cause_label": cause_id if is_anomaly else 0,
                "target_label": target_id if is_anomaly else 0,
                "cause_name": labels["cause"] if is_anomaly else "normal",
                "target_device": labels["target_device"] if is_anomaly else "none",
                "target_interface": labels["target_interface"] if is_anomaly else "",
                "observed_devices": ",".join(sorted(device_frames)),
            }
        )

    return np.stack(scenario_tensor), pd.DataFrame(window_records)


def preprocess_topology_benchmark():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    node_names = build_node_list()
    adjacency = build_adjacency(node_names)

    sample_frame = build_device_frame(
        RAW_ROOT / SCENARIOS[0] / "yang_models" / f"{list_scenario_devices(RAW_ROOT / SCENARIOS[0])[0]}.zip"
    )
    base_feature_names = [column for column in sample_frame.columns if column != "bin_start"]
    feature_names = base_feature_names + ["observation_mask"]

    tensors = []
    metadata_frames = []
    for scenario_name in SCENARIOS:
        scenario_tensor, metadata = build_scenario_tensor(scenario_name, node_names, feature_names)
        tensors.append(scenario_tensor)
        metadata_frames.append(metadata)

    X = np.concatenate(tensors, axis=0)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    y_anomaly = metadata["anomaly_label"].to_numpy(dtype=np.int64)
    y_cause = metadata["cause_label"].to_numpy(dtype=np.int64)
    y_target = metadata["target_label"].to_numpy(dtype=np.int64)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_cause,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_cause[train_idx],
    )

    scaler = StandardScaler()
    X_train_flat = X[train_idx].reshape(-1, X.shape[-1])
    scaler.fit(X_train_flat)

    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    np.save(OUTPUT_ROOT / "X_topology.npy", X_scaled)
    np.save(OUTPUT_ROOT / "y_topology_anomaly.npy", y_anomaly)
    np.save(OUTPUT_ROOT / "y_topology_cause.npy", y_cause)
    np.save(OUTPUT_ROOT / "y_topology_target.npy", y_target)
    np.save(OUTPUT_ROOT / "adjacency.npy", adjacency)
    np.save(OUTPUT_ROOT / "train_idx.npy", train_idx)
    np.save(OUTPUT_ROOT / "val_idx.npy", val_idx)
    np.save(OUTPUT_ROOT / "test_idx.npy", test_idx)
    metadata.to_csv(OUTPUT_ROOT / "window_metadata.csv", index=False)

    joblib.dump(scaler, OUTPUT_ROOT / "scaler.joblib")
    (OUTPUT_ROOT / "node_names.json").write_text(json.dumps(node_names, indent=2))
    (OUTPUT_ROOT / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (OUTPUT_ROOT / "label_maps.json").write_text(
        json.dumps(
            {
                "cause": {idx: name for idx, name in enumerate(CAUSE_ORDER)},
                "target": {idx: name for idx, name in enumerate(TARGET_DEVICE_ORDER)},
            },
            indent=2,
        )
    )

    print(f"Saved topology benchmark with shape {X_scaled.shape}")
    print(metadata.groupby('cause_name').size().to_string())


if __name__ == "__main__":
    preprocess_topology_benchmark()
