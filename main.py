from lib.wbs import WBS
from lib.gantt_chart import GanttChart
from pathlib import Path

RESULTS_DIR = Path("./results")
DATA_DIR = Path("./data").absolute()

def main():

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

    for version in ["v1"]:
        WBS(
            input_path=DATA_DIR / f"wbs_data_{version}.json", 
            output_path_dir=RESULTS_DIR / f"visualization_{version}",
        ).build_wbs()

        GanttChart(
            input_path=DATA_DIR / f"wbs_data_{version}.json",
            output_path_dir=RESULTS_DIR / f"visualization_{version}",
        ).build_gantt_chart(
            draw_dependencies=True,
            draw_groups=True,
        )


if __name__ == "__main__":
    main()
