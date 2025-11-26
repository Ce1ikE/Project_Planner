
# WBS & Gantt Chart Program

Small utility to parse a hierarchical WBS (JSON), flatten it for a Gantt chart and produce visualizations (Gantt chart and WBS graph).

## Project layout
- `main.py` — simple runner that builds the WBS and Gantt visualizations for the data files in `data/`.
- `data/` — contains `wbs_data_v1.json` (sample) and CSV exports generated automatically.
- `lib/` — library code: `gantt_chart.py` (Gantt parser + plot) and `wbs.py` (WBS flatten + visualization).
- `results/` — default output folder for generated PNG / SVG / PDF visualizations.

## JSON / WBS data

Primary file: `data/wbs_data_v1.json`.

Top-level fields used by the code:
- `title` — chart title
- `team`, `description`, `project_scope`, `deliverables`, `acceptance_criteria`, `phases` — optional metadata
- `wbs` — required hierarchical dictionary defining the project tree and tasks

Per-task allowed keys (each task is a dict):
- `start_date` (YYYY-MM-DD)
- `end_date` (YYYY-MM-DD)
- `resources` (list of names)
- `costs` (numeric)
- `dependencies` (list of strings, format: `Group:Task` or `Task`)
- `detailed_description` (string)

Only nodes that include scheduling info (`start_date` or `end_date`) will be included as schedulable tasks in the Gantt chart; groups without dates are treated as parents.

## Automatic CSV export
- When a JSON is loaded for Gantt or WBS processing, the library writes a CSV export next to the JSON file with the same stem (for example `wbs_data_v1.csv`) for quick inspection and import into other tools.

Files created automatically:
- `data/gantt_data_{filename}.csv` — flattened tasks for the Gantt
- `data/wbs_data_{filename}.csv` — parent/child edges 

## Running the project

Recommended Python: 3.11 or newer (see `pyproject.toml`).

Windows PowerShell (recommended workflow):

1) Create and activate a virtual environment

```powershell
python -m venv .venv
# or using uv
# install uv : https://docs.astral.sh/uv/getting-started/installation/
uv venv

# activate the virtual environment (depending on your system)
.\.venv\Scripts\Activate.ps1
# or
.\.venv\Scripts\Activate
# etc..

# If PowerShell blocks activation, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

2) Install required packages:

```powershell
pip install -r requirement.txt
# or using uv
uv sync
# or
uv add -r requirements.txt

```

3) Run the main program `main.py` (entrypoint)

```powershell
# run with plain Python
python .\main.py
# might also be python3 or py (alias in windows)
# or using uv
uv run .\main.py
```

