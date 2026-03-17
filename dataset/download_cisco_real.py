from pathlib import Path

import requests


DATASET_URLS = {
    "baseline.csv.gz": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/0/baseline_no_anomaly.csv.gz",
    "baseline_header.txt": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/0/baseline_header.txt",
    "bgpclear.csv.zip": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/2/bgpclear.csv.zip",
    "bgpclear_ground_truth.txt": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/2/bgpclear_ground_truth.txt",
    "portflap.csv.gz": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/4/portflaptypes.csv.gz",
    "portflap_casedata.txt": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/4/portflap_casedata.txt",
    "portflap_header.txt": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/4/portflap_header.txt",
    "transceiver_pull.csv.zip": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/10/transceiver_pull_20180413_35m.csv.zip",
    "transceiver_pull_event_key.png": "https://raw.githubusercontent.com/cisco-ie/telemetry/master/10/transceiver_pull_event_key.png",
}


def download_file(url, destination):
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"Downloaded {destination.name} ({destination.stat().st_size} bytes)")


def main():
    output_dir = Path("dataset/cisco_real")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in DATASET_URLS.items():
        destination = output_dir / filename
        if destination.exists():
            print(f"Skipping {filename}; already present.")
            continue
        download_file(url, destination)


if __name__ == "__main__":
    main()
