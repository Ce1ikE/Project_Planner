from lib.wbs import WBS
from lib.gantt_chart import GanttChart
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR = RESULTS_DIR.absolute()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR = DATA_DIR.absolute()

def main():

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

    INPUT_FILE = DATA_DIR / "project_data_v1.json"

    OUTPUT_DIR = RESULTS_DIR / "visualization_v1"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    WBS(
        input_path=INPUT_FILE, 
        output_path_dir=OUTPUT_DIR,
    ).build_wbs()

    GanttChart(
        input_path=INPUT_FILE,
        output_path_dir=OUTPUT_DIR,
    ).build_gantt_chart(
        draw_dependencies=True,
        draw_groups=True,
    )


if __name__ == "__main__":
    main()
