from pathlib import Path

import requests


GITHUB_API_ROOT = "https://api.github.com/repos/cisco-ie/telemetry/contents/12"
RAW_ROOT = "https://raw.githubusercontent.com/cisco-ie/telemetry/master/12"

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


def fetch_directory_listing(scenario_name):
    response = requests.get(f"{GITHUB_API_ROOT}/{scenario_name}", timeout=180)
    response.raise_for_status()
    return response.json()


def download_file(url, destination):
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"Downloaded {destination}")


def main():
    root_dir = Path("dataset/cisco_topology_benchmark/raw")
    root_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in SCENARIOS:
        scenario_dir = root_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for metadata_name in ["events.csv", "cdp_map.json", "notes.md", "parent_dataset.json"]:
            destination = scenario_dir / metadata_name
            if destination.exists():
                continue

            url = f"{RAW_ROOT}/{scenario_name}/{metadata_name}"
            response = requests.get(url, timeout=180)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            destination.write_bytes(response.content)
            print(f"Downloaded {destination}")

        for entry in fetch_directory_listing(scenario_name):
            if entry["type"] != "dir" or entry["name"] != "yang_models":
                continue

            yang_dir = scenario_dir / "yang_models"
            yang_dir.mkdir(parents=True, exist_ok=True)

            response = requests.get(entry["url"], timeout=180)
            response.raise_for_status()
            for yang_entry in response.json():
                if yang_entry["type"] != "file" or not yang_entry["name"].endswith(".zip"):
                    continue

                destination = yang_dir / yang_entry["name"]
                if destination.exists():
                    continue
                download_file(yang_entry["download_url"], destination)


if __name__ == "__main__":
    main()
