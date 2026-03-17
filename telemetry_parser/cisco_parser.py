import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "active-routes-count",
    "backup-routes-count",
    "bytes-received",
    "bytes-sent",
    "carrier-transitions",
    "checksum-error-packets",
    "crc-errors",
    "discard-packets",
    "global__established-neighbors-count-total",
    "global__neighbors-count-total",
    "global__nexthop-count",
    "input-data-rate",
    "input-drops",
    "input-errors",
    "input-packet-rate",
    "output-data-rate",
    "output-drops",
    "output-errors",
    "output-packet-rate",
    "packets-received",
    "packets-sent",
    "paths-count",
    "peak-input-data-rate",
    "peak-output-data-rate",
    "total-cpu-five-minute",
    "total-cpu-one-minute",
    "total-number-of-drop-packets",
    "vrf__neighbors-count",
    "vrf__path-count",
    "vrf__update-messages-received",
]

WINDOW_SIZE = 12
WINDOW_STEP = 12

SCENARIO_CONFIG = {
    "baseline": {
        "path": Path("dataset/cisco_real/baseline.csv.gz"),
        "kind": "gzip",
        "cause_id": 0,
    },
    "bgp_clear": {
        "path": Path("dataset/cisco_real/bgpclear.csv.zip"),
        "kind": "zip",
        "cause_id": 1,
    },
    "port_flap": {
        "path": Path("dataset/cisco_real/portflap.csv.gz"),
        "kind": "gzip",
        "cause_id": 2,
    },
    "transceiver_pull": {
        "path": Path("dataset/cisco_real/transceiver_pull.csv.zip"),
        "kind": "zip",
        "cause_id": 3,
    },
}

# These event keys are read directly from Cisco's transceiver pull event chart.
TRANSCEIVER_EVENT_STARTS_NS = [
    1523638508213000000,
    1523638878096000000,
    1523639245860000000,
    1523639561045000000,
    1523639884947000000,
]
TRANSCEIVER_EVENT_DURATION_NS = 30_000_000_000


def load_bgp_intervals():
    ground_truth = pd.read_csv("dataset/cisco_real/bgpclear_ground_truth.txt")
    return [
        (int(row["Start"]), int(row["End"]))
        for _, row in ground_truth.iterrows()
    ]


def load_port_flap_intervals():
    los_angeles = ZoneInfo("America/Los_Angeles")
    intervals = []
    lines = Path("dataset/cisco_real/portflap_casedata.txt").read_text().splitlines()

    for line in lines:
        if "|" not in line or "Port  Flap" in line or "Clean Base" in line or "____" in line:
            continue

        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) < 10 or not parts[3] or not parts[4]:
            continue

        start = (
            datetime.strptime(parts[3], "%m/%d/%y %H:%M")
            .replace(tzinfo=los_angeles)
            .astimezone(ZoneInfo("UTC"))
            .timestamp()
        )
        end = (
            datetime.strptime(parts[4], "%m/%d/%y %H:%M")
            .replace(tzinfo=los_angeles)
            .astimezone(ZoneInfo("UTC"))
            .timestamp()
        )
        intervals.append((int(start), int(end)))

    return intervals


def interval_contains(intervals, value_sec):
    return any(start <= value_sec < end for start, end in intervals)


def transceiver_contains(value_sec):
    value_ns = value_sec * 1_000_000_000
    return any(
        start <= value_ns < start + TRANSCEIVER_EVENT_DURATION_NS
        for start in TRANSCEIVER_EVENT_STARTS_NS
    )


def load_raw_scenario(path, kind):
    columns = ["time"] + FEATURES
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            with archive.open(csv_name) as csv_file:
                frame = pd.read_csv(
                    csv_file,
                    usecols=lambda name: name in columns,
                    low_memory=False,
                )
    else:
        frame = pd.read_csv(
            path,
            compression="gzip",
            usecols=lambda name: name in columns,
            low_memory=False,
        )

    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    frame["time_sec"] = (frame["time"] // 1_000_000_000).astype("int64")
    frame["bin_start"] = (frame["time_sec"] // 5) * 5

    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame.get(feature, 0), errors="coerce").fillna(0.0)

    aggregated = frame.groupby("bin_start")[FEATURES].agg(["mean", "max"]).sort_index()
    aggregated.columns = [f"{feature}_{aggregation}" for feature, aggregation in aggregated.columns]
    return aggregated.reset_index()


def label_scenario(frame, scenario_name, bgp_intervals, port_intervals):
    if scenario_name == "baseline":
        frame["is_anomaly"] = 0
        frame["rca_label"] = 0
    elif scenario_name == "bgp_clear":
        frame["is_anomaly"] = frame["bin_start"].apply(
            lambda value: int(interval_contains(bgp_intervals, value))
        )
        frame["rca_label"] = np.where(frame["is_anomaly"] == 1, 1, 0)
    elif scenario_name == "port_flap":
        frame["is_anomaly"] = frame["bin_start"].apply(
            lambda value: int(interval_contains(port_intervals, value))
        )
        frame["rca_label"] = np.where(frame["is_anomaly"] == 1, 2, 0)
    elif scenario_name == "transceiver_pull":
        frame["is_anomaly"] = frame["bin_start"].apply(lambda value: int(transceiver_contains(value)))
        frame["rca_label"] = np.where(frame["is_anomaly"] == 1, 3, 0)
    else:
        raise ValueError(f"Unsupported scenario: {scenario_name}")

    frame["scenario"] = scenario_name
    return frame


def create_windows(aggregated_frame, feature_columns):
    X_windows = []
    y_anomaly = []
    y_rca = []
    metadata = []

    for scenario_name, group in aggregated_frame.groupby("scenario", sort=False):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        anomaly_values = group["is_anomaly"].to_numpy(dtype=np.int64)
        rca_values = group["rca_label"].to_numpy(dtype=np.int64)
        bins = group["bin_start"].to_numpy(dtype=np.int64)

        for start_index in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_STEP):
            end_index = start_index + WINDOW_SIZE
            window_anomalies = anomaly_values[start_index:end_index]
            window_rca = rca_values[start_index:end_index]

            X_windows.append(values[start_index:end_index])
            y_anomaly.append(int(window_anomalies.max() > 0))
            y_rca.append(int(window_rca.max()))
            metadata.append(
                {
                    "scenario": scenario_name,
                    "window_start": int(bins[start_index]),
                    "window_end": int(bins[end_index - 1]),
                    "window_label": int(window_rca.max()),
                }
            )

    return (
        np.stack(X_windows),
        np.array(y_anomaly, dtype=np.int64),
        np.array(y_rca, dtype=np.int64),
        pd.DataFrame(metadata),
    )


def preprocess_cisco_telemetry():
    output_dir = Path("dataset/cisco_real_processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    bgp_intervals = load_bgp_intervals()
    port_intervals = load_port_flap_intervals()

    scenario_frames = []
    for scenario_name, config in SCENARIO_CONFIG.items():
        raw_frame = load_raw_scenario(config["path"], config["kind"])
        labeled_frame = label_scenario(raw_frame, scenario_name, bgp_intervals, port_intervals)
        scenario_frames.append(labeled_frame)

    aggregated = pd.concat(scenario_frames, ignore_index=True)
    feature_columns = [
        column
        for column in aggregated.columns
        if column.endswith("_mean") or column.endswith("_max")
    ]

    scaler = StandardScaler()
    aggregated[feature_columns] = scaler.fit_transform(aggregated[feature_columns])
    joblib.dump(scaler, output_dir / "cisco_scaler.joblib")

    X, y_anomaly, y_rca, metadata = create_windows(aggregated, feature_columns)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=y_rca,
    )

    np.save(output_dir / "X_cisco.npy", X)
    np.save(output_dir / "y_cisco_anomaly.npy", y_anomaly)
    np.save(output_dir / "y_cisco_rca.npy", y_rca)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "test_idx.npy", test_idx)
    metadata.to_csv(output_dir / "cisco_window_metadata.csv", index=False)
    aggregated.to_csv(output_dir / "cisco_aggregated.csv", index=False)

    print(f"Saved Cisco aggregated bins to {output_dir / 'cisco_aggregated.csv'}")
    print(f"Saved Cisco windows: X={X.shape}, anomaly positives={int(y_anomaly.sum())}")
    print(
        "RCA class counts:",
        {
            int(label): int(count)
            for label, count in zip(*np.unique(y_rca, return_counts=True))
        },
    )


if __name__ == "__main__":
    preprocess_cisco_telemetry()
