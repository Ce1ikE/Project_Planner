from enum import Enum
import matplotlib.pyplot as plt

class ProjectKeys(Enum):
    """Keys for top-level project information"""
    TITLE = "title"
    WBS = "wbs"
    PHASES = "phases"
    ORGANOGRAM = "organogram"
    PROJECT_INFO = "project_info"

class IgnoredKeys(Enum):
    """Keys to ignore when flattening structures"""
    _DESCRIPTION = "_description"
    _NOTES = "_notes"
    _COMMENT = "_comment"

IGNORED_KEYS = {key.value for key in IgnoredKeys}

class GanttChartKeys(Enum):
    """Keys for Gantt chart task information in WBS"""
    ROOT = ProjectKeys.WBS.value
    TASK_NAME = "task_name"
    START_DATE = "start_date"
    END_DATE = "end_date"
    DURATION = "duration"
    RESOURCES = "resources"
    COSTS = "costs"
    DEPENDENCIES = "dependencies"
    DELIVERABLES = "deliverables"
    DETAILED_DESCRIPTION = "detailed_description"
    PARENT = "parent"
    CHILDREN = "children"

class WBSKeys(Enum):
    """Keys for WBS information"""
    ROOT = ProjectKeys.WBS.value
    NAME = "name"
    PARENT = "parent"
    CHILDREN = "children"

class OrganogramKeys(Enum):
    """Keys for Organogram chart information"""
    ROOT = ProjectKeys.ORGANOGRAM.value
    NAME = "name"
    ROLE = "role"
    SUPERVISOR_OF = "supervisor_of"
    REPORTS_TO = "reports_to"

class GlobalConst:
    """Global constants for visualization styling"""
    DEFAULT_WBS_TITLE = "WBS Visualization"
    DEFAULT_GANTT_TITLE = "Gantt Chart Visualization"
    DEFAULT_ORGANOGRAM_TITLE = "Organogram Chart Visualization"
    DEFAULT_MILESTONE_TITLE = "Milestone Chart Visualization"
    DEFAULT_DELIVERABLES_TITLE = "Deliverables Table Visualization"

    DEFAULT_MILESTONE_COLUMN_PHASE = "Phase"
    DEFAULT_MILESTONE_COLUMN_GROUP = "Group"
    DEFAULT_MILESTONE_COLUMN_DATE = "Date"

    DEFAULT_DELIVERABLES_COLUMN_PHASE = "Phase"
    DEFAULT_DELIVERABLES_COLUMN_GROUP = "Group"
    DEFAULT_DELIVERABLES_COLUMN_DELIVERABLES = "Deliverable"

    DEFAULT_SAVEDPI = 300

    TITLE_SIZE = 14
    TITLE_FONT_WEIGHT = "bold"

    DATE_FORMAT = '%Y-%m-%d'

    FONT_COLOR = "#6C6C6C"
    
    X_LABEL = "Timeline"
    Y_LABEL = "Tasks"
    LABEL_SIZE = 8

    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    # qualitative colormap
    COLOR_MAP_GROUPS = plt.get_cmap("tab20")
    # qualitative colormap
    COLOR_MAP_ROLES = plt.get_cmap("Pastel1")
    
    DAY_FONT_SIZE = 8
    DAY_FONT_WEIGHT = "bold"
    DAY_FONT_COLOR = FONT_COLOR
    
    MONTH_FONT_SIZE = 10
    MONTH_FONT_WEIGHT = "bold"
    MONTH_FONT_COLOR = FONT_COLOR
    
    DEPENDENCIES_COLOR = "#767170"
    
    GROUP_ALPHA = 0.2
    
    FONT_DICT = {
        "fontfamily": "monospace",
        "fontsize": 10,
        "fontweight": "bold",
        "color": FONT_COLOR,
    }
    
    RANDOM_SEED = 5106

    TERMINAL_COLORS = {
        "RED"         : "\033[38;5;196m",
        "GREEN"       : "\033[38;5;119m",
        "BLUE"        : "\033[38;5;21m",
        "VIOLET"      : "\033[38;5;129m",
        "PURPLE"      : "\033[38;5;90m",
        "PINK"        : "\033[38;5;198m",
        "CYAN"        : "\033[38;5;87m",
        "ORANGE"      : "\033[38;5;202m",
        "YELLOW"      : "\033[38;5;226m",
        "GOLD"        : "\033[38;5;172m",
        "TURQUOISE"   : "\033[38;5;37m",
        "RESET_COLOR" : "\033[0m",
    }


