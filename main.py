from lib.gantt_chart import GanttBuilder
from lib.organogram import OrganogramBuilder
from pathlib import Path
from scripts.robot_analysis import run

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

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file {INPUT_FILE} does not exist.")

    OUTPUT_DIR = RESULTS_DIR / "visualization_v1"

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    GanttBuilder(
        input_path=INPUT_FILE,
        output_path_dir=OUTPUT_DIR,
    ).build_milestone_chart_table(
        draw_groups=True,
        base_row_height=0.2,
        headers=[
            "Fase",
            "Datum",
        ],
    ).build_deliverables_table(
        draw_groups=True,
        base_row_height=0.2,
        headers=[
            "Fase",
            "Deliverable",
        ],
    ).build_gantt_chart(
        draw_dependencies=True,
        draw_groups=True,
        max_label_size=50,
    ).build_wbs(
        draw_groups=True,
    )

    OrganogramBuilder(
        input_path=INPUT_FILE,
        output_path_dir=OUTPUT_DIR,
    ).build_organogram()


    run(
        OUTPUT_DIR, DATA_DIR / "cost_data.csv"
    )





if __name__ == "__main__":
    main()
