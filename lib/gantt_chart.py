import json
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import datetime as dt

from pathlib import Path
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch

from .global_const import *
from .util_functions import *

class GanttBuilder:
    def __init__(
        self,
        input_path: Path = None,
        output_path_dir: Path = None,
        data_date_format: str = None,
        title_size: int = None,
        title_font_weight: int = None,
        y_label = None,
        x_label = None,
        day_font_size = None,
        day_font_weight = None,
        day_font_color = None,
        month_font_size = None,
        month_font_weight = None,
        month_font_color = None,
        dependencies_color = None,
        group_alpha = None,
        font_dict: dict = None,
        color_map = None,
    ):
        """
        PlanningBuilder class for visualization of project schedules.
        Allows building charts from hierarchical data: 
            - Gantt charts
            - Milestone tables
            - Deliverables tables
            - WBS diagrams 


        Args:
            input_path (Path): Path to the input file containing task data.
            output_path_dir (Path): Directory where output files will be saved.
            data_date_format (str): Format of the date strings in the input data.
            title_size (int): Font size for the chart title.
            title_font_weight (int): Font weight for the chart title.
            y_label (str): Label for the Y-axis.
            x_label (str): Label for the X-axis.
            day_font_size (int): Font size for day labels.
            day_font_weight (int): Font weight for day labels.
            day_font_color (str): Font color for day labels.
            month_font_size (int): Font size for month labels.
            month_font_weight (int): Font weight for month labels.
            month_font_color (str): Font color for month labels.
            dependencies_color (str): Color for dependency lines.
            group_alpha (float): Transparency level for group bars.
            font_dict (dict): Font properties for text elements.
            color_map: Color map for groups.
        """

        self.input_path = input_path

        self.output_path_dir = output_path_dir
        self.output_path_dir.mkdir(parents=True, exist_ok=True)
        if self.output_path_dir.exists() is False:
            raise ValueError("Output path directory must exist.")
        
        self.y_label = y_label if y_label else GlobalConst.Y_LABEL
        self.x_label = x_label if x_label else GlobalConst.X_LABEL
        
        self.day_font_size = day_font_size if day_font_size else GlobalConst.DAY_FONT_SIZE
        self.day_font_weight = day_font_weight if day_font_weight else GlobalConst.DAY_FONT_WEIGHT
        self.day_font_color = day_font_color if day_font_color else GlobalConst.DAY_FONT_COLOR
        
        self.title_size = title_size if title_size else GlobalConst.TITLE_SIZE
        self.title_font_weight = title_font_weight if title_font_weight else GlobalConst.TITLE_FONT_WEIGHT

        self.month_font_size = month_font_size if month_font_size else GlobalConst.MONTH_FONT_SIZE
        self.month_font_weight = month_font_weight if month_font_weight else GlobalConst.MONTH_FONT_WEIGHT
        self.month_font_color = month_font_color if month_font_color else GlobalConst.MONTH_FONT_COLOR

        self.dependencies_color = dependencies_color if dependencies_color else GlobalConst.DEPENDENCIES_COLOR
        
        self.group_alpha = group_alpha if group_alpha else GlobalConst.GROUP_ALPHA
        
        self.fontdict = font_dict if font_dict else GlobalConst.FONT_DICT
        
        self.data_date_format = data_date_format if data_date_format else GlobalConst.DATE_FORMAT
        
        self.color_map = color_map if color_map else GlobalConst.COLOR_MAP_GROUPS

        self.tasks_df: pd.DataFrame = None
        self.wbs_df: pd.DataFrame = None

        if self.input_path:
            if self.input_path.suffix == '.json':
                self._from_json(self.input_path)
                # YAML is also a possibility in the future or other formats that support hierarchical data (like XML)
            else:
                raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        
        if self.tasks_df is None or self.tasks_df.empty or self.wbs_df is None or self.wbs_df.empty:
            raise ValueError("No data available to build charts.")
        
        self._clean_data()

    def _clean_data(self):
        # convert date columns to datetime
        self.tasks_df[GanttChartKeys.START_DATE.value] = pd.to_datetime(
            self.tasks_df[GanttChartKeys.START_DATE.value], 
            format=self.data_date_format
        )
        self.tasks_df[GanttChartKeys.END_DATE.value] = pd.to_datetime(
            self.tasks_df[GanttChartKeys.END_DATE.value], 
            format=self.data_date_format
        )
        # compute duration in days
        self.tasks_df[GanttChartKeys.DURATION.value] = (
            self.tasks_df[GanttChartKeys.END_DATE.value] - self.tasks_df[GanttChartKeys.START_DATE.value]
        ).dt.days
        # fill NaN in resources and costs
        self.tasks_df.fillna(
            {
                GanttChartKeys.RESOURCES.value: {}, 
                GanttChartKeys.COSTS.value: 0
            }, 
            inplace=True
        )

    @time_decorator
    def _from_xml(self, file_path: Path):
        raise NotImplementedError("XML input format not yet implemented.")

    @time_decorator
    def _from_yaml(self, file_path: Path):
        raise NotImplementedError("YAML input format not yet implemented.")

    @time_decorator
    def _from_json(self, file_path: Path):
        """
        Loads task data from a JSON file and flattens the hierarchical structure.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        tasks, wbs = self._flatten_wbs(data)
        
        self.tasks_df = pd.DataFrame(tasks)
        self.tasks_df.to_csv(file_path.parent / f'gantt_data_{file_path.stem}.csv', index=False)
        self.tasks_df.info()

        self.wbs_df = pd.DataFrame(wbs) 
        self.wbs_df.to_csv(file_path.parent / f'wbs_data_{file_path.stem}.csv', index=False)
        self.wbs_df.info()

    def _flatten_wbs(self, data: dict, parent=None, level=0, rows_tasks=None, rows_wbs=None):
        """
        Flattens a hierarchical WBS structure into a list of tasks with scheduling info.
        Args:
            data (dict): The WBS data passed recursively for each node.
            parent (str): The parent task name.
            level (int): Current level in the hierarchy.
            rows_tasks (list): Accumulator for flattened tasks.
            rows_wbs (list): Accumulator for WBS structure.
        """

        # at first entry of the recursion 
        # we initialize the rows list of tasks and wbs
        if level == 0:
            rows_tasks = []
            rows_wbs = []
            # preserve the top-level title if present
            self.title = data.get(ProjectKeys.TITLE.value, None)
            root_wbs: dict = data.get(ProjectKeys.WBS.value, None)
            
            if root_wbs is None:
                raise ValueError(f"No root key {ProjectKeys.WBS.value} found in the input data.")

            data = root_wbs

        for key, value in data.items():
            if key in IGNORED_KEYS:
                continue

            if isinstance(value, dict):
                rows_wbs.append({
                    WBSKeys.PARENT.value: parent if parent else ProjectKeys.WBS.value,
                    WBSKeys.CHILDREN.value: key,
                })
                # leaf task (has scheduling info)
                if GanttChartKeys.START_DATE.value in value and GanttChartKeys.END_DATE.value in value:
                    rows_tasks.append({
                        GanttChartKeys.TASK_NAME.value: key,
                        GanttChartKeys.START_DATE.value: value.get(GanttChartKeys.START_DATE.value, None),
                        GanttChartKeys.END_DATE.value: value.get(GanttChartKeys.END_DATE.value, None),
                        GanttChartKeys.RESOURCES.value: value.get(GanttChartKeys.RESOURCES.value, []),
                        GanttChartKeys.COSTS.value: value.get(GanttChartKeys.COSTS.value, 0),
                        GanttChartKeys.DEPENDENCIES.value: value.get(GanttChartKeys.DEPENDENCIES.value, []),
                        GanttChartKeys.DELIVERABLES.value: value.get(GanttChartKeys.DELIVERABLES.value, []),
                        GanttChartKeys.DETAILED_DESCRIPTION.value: value.get(GanttChartKeys.DETAILED_DESCRIPTION.value, ""),
                        GanttChartKeys.PARENT.value: parent if parent else None,
                        GanttChartKeys.CHILDREN.value: key,
                    })
                else:
                    # group node: recurse into its children and pass current key as parent
                    self._flatten_wbs(value, key, level + 1, rows_tasks, rows_wbs)

        return rows_tasks, rows_wbs
    
    def _create_group_color_map(self, draw_groups: bool):
        if not draw_groups:
            return {}

        group_names = self.tasks_df[GanttChartKeys.PARENT.value].dropna().unique()
        group_color_map = {
            name: self.color_map(i % self.color_map.N) 
            for i, name in enumerate(group_names)
        }
        return group_color_map
    
    @time_decorator
    def build_milestone_chart_table(
        self, 
        draw_groups: bool = False,
        base_row_height: float = 0.2,
        headers: list = None,
    ):
        """
        Build a milestone chart table from the tasks dataframe.

        Args:
            draw_groups (bool): Whether to color rows based on their group.
            base_row_height (float): Base height of each row in the table.
        """  
        headers = headers if headers and 2 >= len(headers) > 0 else [
            GlobalConst.DEFAULT_MILESTONE_COLUMN_PHASE,
            GlobalConst.DEFAULT_MILESTONE_COLUMN_DATE,
        ]

        milestone_rows = []
        for idx, task in self.tasks_df.iterrows():
            milestone_rows.append({
                headers[0]: task[GanttChartKeys.TASK_NAME.value],
                GlobalConst.DEFAULT_MILESTONE_COLUMN_GROUP: task[GanttChartKeys.PARENT.value] if draw_groups else None,
                headers[1]: task[GanttChartKeys.END_DATE.value].strftime(self.data_date_format) if pd.notna(task[GanttChartKeys.END_DATE.value]) else None
            })

        milestone_df = pd.DataFrame(milestone_rows)
        milestone_df.to_csv(self.output_path_dir / 'milestone_chart_data.csv', index=False)
        
        group_color_map = self._create_group_color_map(draw_groups)

        fig, ax = plt.subplots(figsize=(5, (len(milestone_df) + 1) * base_row_height * 1.2))
        ax.axis('off')
        ax.set_title(
            self.title + "\n" + GlobalConst.DEFAULT_MILESTONE_TITLE if self.title else GlobalConst.DEFAULT_MILESTONE_TITLE, 
            fontdict=self.fontdict
        )

        df_to_display = milestone_df.drop(columns=[GlobalConst.DEFAULT_MILESTONE_COLUMN_GROUP])
        col_labels = df_to_display.columns.tolist()
        cell_text = df_to_display.values.tolist()

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc='left',
            loc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(self.fontdict.get("fontsize", 10))
        table.auto_set_column_width(col=list(range(len(col_labels))))

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("none") 
            cell.set_edgecolor(self.fontdict.get("color", "black"))
            
            cell.set_text_props(
                ha='left', 
                va='center', 
                wrap=False,
                **self.fontdict
            )

            if row == 0:
                cell.set_facecolor('#f2f2f2')
                cell.set_text_props(fontweight=self.fontdict.get('fontweight', 'bold'))
            else:
                task_data = milestone_df.iloc[row - 1]
                
                # --- COLORING LOGIC ---
                if draw_groups:
                    group_name = task_data[GlobalConst.DEFAULT_MILESTONE_COLUMN_GROUP]
                    if group_name and group_name in group_color_map:
                        color = group_color_map[group_name]
                        cell.set_facecolor(list(color[:3]) + [self.group_alpha])

        plt.tight_layout(pad=2)
        plt.savefig(self.output_path_dir / 'milestone_table.png', dpi=GlobalConst.DEFAULT_SAVEDPI)
        plt.savefig(self.output_path_dir / 'milestone_table.svg', format='svg')
        plt.savefig(self.output_path_dir / 'milestone_table.pdf', format='pdf')
        plt.close()

        return self
    
    @time_decorator
    def build_deliverables_table(
        self, 
        draw_groups: bool = False,
        base_row_height: float = 0.25,
        headers: list = None,
    ):
        """
        Build a deliverables table from the tasks dataframe.

        Args:
            draw_groups (bool): Whether to color rows based on their group.
            base_row_height (float): Base height of each row in the table.
            headers (list): List of column headers for the table.
        """

        headers = headers if headers and 2 >= len(headers) > 0 else [
            GlobalConst.DEFAULT_DELIVERABLES_COLUMN_PHASE,
            GlobalConst.DEFAULT_DELIVERABLES_COLUMN_DELIVERABLES,
        ]

        deliverable_rows = []
        for idx, task in self.tasks_df.iterrows():
            deliverables = task[GanttChartKeys.DELIVERABLES.value]

            if not isinstance(deliverables, (list, tuple)):
                deliverables = [deliverables] if pd.notna(deliverables) and deliverables else []
            
            wrapper = textwrap.TextWrapper(width=50)
            joined_deliverable = '\n'.join([wrapper.fill(str(d)) for d in deliverables])

            deliverable_rows.append({
                headers[0]: task[GanttChartKeys.TASK_NAME.value],
                GlobalConst.DEFAULT_DELIVERABLES_COLUMN_GROUP: task[GanttChartKeys.PARENT.value] if draw_groups else None,
                headers[1]: joined_deliverable
            })

        deliverables_df = pd.DataFrame(deliverable_rows)
        deliverables_df.to_csv(self.output_path_dir / 'deliverables_table.csv', index=False)

        group_color_map = self._create_group_color_map(draw_groups)
        
        max_lines = 1
        for text in deliverables_df[GlobalConst.DEFAULT_DELIVERABLES_COLUMN_DELIVERABLES]:
            max_lines = max(max_lines, str(text).count('\n') + 1)


        fig_height = (len(deliverables_df) + 1) * base_row_height * 0.8 * max_lines
        fig, ax = plt.subplots(figsize=(10, fig_height)) 
        ax.axis('off')
        ax.set_title(
            self.title + "\n" + GlobalConst.DEFAULT_DELIVERABLES_TITLE if self.title else GlobalConst.DEFAULT_DELIVERABLES_TITLE, 
            fontdict=self.fontdict
        )

        df_to_display = deliverables_df.drop(columns=[GlobalConst.DEFAULT_DELIVERABLES_COLUMN_GROUP])
        col_labels = df_to_display.columns.tolist()
        cell_text = df_to_display.values.tolist()

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            bbox=[0, 0, 1, 1],
            cellLoc='left',
            rowLoc='top',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(self.fontdict.get("fontsize", 10))
        table.auto_set_column_width(col=list(range(len(col_labels))))

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("none") 
            cell.set_edgecolor(self.fontdict.get("color", "black"))
            
            cell.set_text_props(
                wrap=True,
                **self.fontdict
            )

            if row == 0:
                cell.set_facecolor('#f2f2f2')
                cell.set_text_props(fontweight=self.fontdict.get('fontweight', 'bold'))
                cell.set_height(base_row_height * 1.5)
            else:
                task_data = deliverables_df.iloc[row - 1]
                
                # --- COLORING LOGIC ---
                if draw_groups:
                    group_name = task_data[GlobalConst.DEFAULT_DELIVERABLES_COLUMN_GROUP]
                    if group_name and group_name in group_color_map:
                        color = group_color_map[group_name]
                        cell.set_facecolor(list(color[:3]) + [self.group_alpha])
                
                # --- DYNAMIC ROW HEIGHT LOGIC ---
                deliverables_text = str(task_data[GlobalConst.DEFAULT_DELIVERABLES_COLUMN_DELIVERABLES])
                num_lines = deliverables_text.count('\n') + 1 
                dynamic_height = base_row_height * num_lines

                for c in range(len(col_labels)):
                    table.get_celld()[(row, c)].set_height(dynamic_height)

        plt.tight_layout(pad=2)
        plt.savefig(self.output_path_dir / 'deliverables_table.png', dpi=GlobalConst.DEFAULT_SAVEDPI)
        plt.savefig(self.output_path_dir / 'deliverables_table.svg', format='svg')
        plt.savefig(self.output_path_dir / 'deliverables_table.pdf', format='pdf')
        plt.close()

        return self

    def _build_week_ticks(self, start_date, end_date):
        mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        return mondays, [d.strftime('%d') for d in mondays]
    
    def _build_annotation_hbox(
        self, 
        ax: plt.Axes, 
        x, 
        y, 
        text, 
        fontsize: int = 8, 
        fontweight: str = 'normal', 
        fontcolor: str = 'black',
        max_width: int = 30
    ):
        wrapper = textwrap.TextWrapper(width=max_width)
        wrapped_text = wrapper.fill(text)

        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            fc="white", 
            ec="black", 
            lw=0.8
        )

        ax.annotate(
            wrapped_text,
            xy=(x, y),
            xytext=(5, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=self.fontdict.get('fontsize', fontsize),
            fontweight=self.fontdict.get('fontweight', fontweight),
            color=self.fontdict.get('color', fontcolor),
            fontfamily=self.fontdict.get('fontfamily', 'monospace'),
            bbox=bbox_props,
            clip_on=False
        )

    @time_decorator
    def build_gantt_chart(
        self, 
        draw_dependencies: bool = False, 
        draw_groups: bool = False, 
        max_label_size: int = 50
    ):
        """
        Args:
            draw_dependencies (bool): Whether to draw dependency arrows between tasks.
            draw_groups (bool): Whether to color bars based on their group.
            max_label_size (int): Maximum size of the label next to bars.
        """


        week_positions, week_labels = self._build_week_ticks(
            self.tasks_df[GanttChartKeys.START_DATE.value].min(), 
            self.tasks_df[GanttChartKeys.END_DATE.value].max()
        )

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_title(self.title + "\n" + GlobalConst.DEFAULT_GANTT_TITLE if self.title else GlobalConst.DEFAULT_GANTT_TITLE, fontdict=self.fontdict)
        ax.tick_params(axis='both')

        # 1 task is a bar with start and end date
        tasks = self.tasks_df.sort_values(
            by=[
                GanttChartKeys.START_DATE.value, 
                # in case of parrallel tasks, sort by name to have consistent order
                GanttChartKeys.TASK_NAME.value
            ], 
            ascending=False
        )
        # foreach task we'll create a bar and add annotations
        # draw bars with automatic color assignment (using a colormap)
        # store geometry for optional dependency drawing
        bar_info = {}
        groups = defaultdict(list)
        group_color_map = self._create_group_color_map(draw_groups)

        for idx, task in enumerate(tasks.itertuples(index=False, name='Task')):
            
            task_name = getattr(task, GanttChartKeys.TASK_NAME.value)
            start: pd.Timestamp =  getattr(task, GanttChartKeys.START_DATE.value)
            end: pd.Timestamp = getattr(task, GanttChartKeys.END_DATE.value)
            parent = getattr(task, GanttChartKeys.PARENT.value) or "Ungrouped"
            duration: int = (end - start).days
            resources = list(getattr(task, GanttChartKeys.RESOURCES.value)) if isinstance(getattr(task, GanttChartKeys.RESOURCES.value), (list, tuple)) else getattr(task, GanttChartKeys.RESOURCES.value)
            costs = getattr(task, GanttChartKeys.COSTS.value)
            color = self.color_map(idx % self.color_map.N)

            bar: plt.Rectangle = ax.barh(
                task_name,
                width=duration,
                height=0.6,
                left=start,
                edgecolor='black',
                linewidth=2.5,
                linestyle='-',
                color=color
            )[0]

            # numeric start/end for transforms and dependency drawing
            try:
                start_num = mdates.date2num(start.to_pydatetime())
            except Exception:
                start_num = mdates.date2num(start)

            try:
                end_num = mdates.date2num(end.to_pydatetime())
            except Exception:
                end_num = mdates.date2num(end)

            y_center = bar.get_y() + bar.get_height() / 2
            x_center = start_num + (duration / 2.0)
            bar_info[task_name] = {
                'y': y_center,
                'x': x_center,
                'bottom': bar.get_y(),
                'left': bar.get_x(),
                'width': bar.get_width(),
                'height': bar.get_height(),
                'start_num': start_num,
                'end_num': end_num,
                'parent': parent
            }
            groups[parent].append(bar_info[task_name])

            resources_str = ', '.join(resources) if resources else ''
            if len(resources_str) > max_label_size:
                resources_str = resources_str[:max_label_size - 3] + '...'
            
            label = f"{duration}d | €{costs}"
            if resources_str:
                label = label + ' | ' + resources_str

            gap = 2.5
            # put text next to the right of the bar
            self._build_annotation_hbox(
                ax,
                x=bar.get_x() + bar.get_width() + gap,
                y=y_center,
                text=label,
                fontsize=self.fontdict.get('fontsize', 8),
                fontweight=self.fontdict.get('fontweight', 'normal'),
                fontcolor=self.fontdict.get('color', 'black'),
                max_width=60
            )

        # draw horizontal lines on top of the graph to group tasks by parent
        if draw_groups:
            
            for group_name, infos in groups.items():
                ys = [i['y'] for i in infos]
                # expand a little so band covers bar height
                top = max([y + 0.4 for y in ys])
                bottom = min([y - 0.4 for y in ys])
                color = group_color_map.get(group_name, (0.9, 0.9, 0.9))

                # adds a horizontal span accross the group
                ax.axhspan(
                    bottom, 
                    top, 
                    xmin=0, 
                    xmax=1, 
                    facecolor=color, 
                    alpha=self.group_alpha, 
                    zorder=0
                )

        # draws orthogonal dependency connectors
        if draw_dependencies:
            for idx, task in enumerate(tasks.itertuples(index=False)):
                deps = task.dependencies if isinstance(task.dependencies, (list, tuple)) else ([task.dependencies] if task.dependencies else [])
                for dep in deps:
                    if not dep:
                        continue
                    # dependency may be in format 'Group:Task' inside the json source file
                    pred_name = dep.split(':', 1)[1].strip() if ':' in dep else dep.strip()
                    if pred_name not in bar_info or task.task_name not in bar_info:
                        continue
                    
                    pred = bar_info[pred_name]
                    succ = bar_info[task.task_name]
                    pred_x = pred['x']
                    pred_y = pred['y']
                    pred_bottom = pred['bottom']
                    pred_start_num = pred['start_num']
                    pred_end_num = pred['end_num']
                    pred_left = pred['left']
                    pred_width = pred['width']
                    pred_right = pred_left + pred_width
                    pred_parent = pred.get('parent', None)
                    
                    succ_x = succ['x']
                    succ_y = succ['y']
                    succ_bottom = succ['bottom']
                    succ_start_num = succ['start_num']
                    succ_end_num = succ['end_num']
                    succ_left = succ['left']
                    succ_width = succ['width']
                    succ_right = succ_left + succ_width
                    succ_parent = succ.get('parent', None)

                    # build orthogonal polyline points if pred_x < succ_x
                    # from   (pred x_center,pred bottom) -> (pred x_center, succ y_center)
                    # from   (pred x_center, succ y_center) -> (succ x_center, succ y_center)
                    x_pts = [pred_x, pred_x, succ_left]
                    y_pts = [pred_bottom, succ_y, succ_y]

                    # if pred_x == succ_x then we need a other way to indicate interdependency  between tasks with similar parent
                    # we'll use markers in the center of each bar to indicate interdependency between tasks with similar start positions
                    if pred_x == succ_x and pred_parent == succ_parent:
                        curr_pred_count = pred.get("dependencies_marker", 0) + 1
                        curr_succ_count = succ.get("dependencies_marker", 0) + 1
                        pred["dependencies_marker"] = curr_pred_count
                        succ["dependencies_marker"] = curr_succ_count
                        
                        if pred.get("dependencies_marker", 0) > 0 or succ.get("dependencies_marker", 0) > 0:
                            clr_dependencies = plt.get_cmap("hsv")
                            clr_dependencies = clr_dependencies(pred["dependencies_marker"] * 1.0 / 12)
                            pred_x += 0.4 * pred.get("dependencies_marker", 0)
                            succ_x += 0.4 * succ.get("dependencies_marker", 0)
                        else:
                            clr_dependencies = self.dependencies_color

                        ax.scatter(
                            [pred_x, succ_x], 
                            [pred_y, succ_y], 
                            s=10, 
                            marker='o', 
                            linewidth=0.5, 
                            facecolor=clr_dependencies,
                            zorder=2,
                            hatch='/' * pred.get("dependencies_marker", 0),
                            edgecolor="black"
                        )
                    else:
                        ax.plot(
                            x_pts, 
                            y_pts, 
                            color=self.dependencies_color, 
                            lw=0.9, 
                            zorder=1
                        )
                        ax.annotate(
                            '', 
                            xy=(succ_left, succ_y), 
                            xytext=(succ_left - 0.01, succ_y),
                            arrowprops=dict(arrowstyle='->', color=self.dependencies_color, lw=1.0), 
                            annotation_clip=False
                        )

        # annotate the first x-axis for the weeks 
        ax.set_xticks(week_positions)
        ax.set_xticklabels(week_labels, fontsize=self.day_font_size, color=self.day_font_color)
        # creates a second x-axis for the months 
        sec_ax = ax.secondary_xaxis('bottom')
        sec_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
        sec_ax.xaxis.set_major_locator(mdates.MonthLocator())
        sec_ax.spines['bottom'].set_position(('outward', 20))
        sec_ax.tick_params(
            axis='x', 
            labelsize=self.month_font_size, 
            colors=self.month_font_color
        )
        # month line formatting
        for label in sec_ax.get_xticklabels():
            label.set_fontsize(self.month_font_size)
            label.set_fontweight(self.month_font_weight)
            label.set_color(self.month_font_color)
            label.set_fontfamily(self.fontdict.get("fontfamily", "monospace"))

        ax.set_xlabel(
            self.x_label, 
            fontdict=self.fontdict,
            labelpad=10.0,
            loc="center"
        )
        ax.set_ylabel(
            self.y_label, 
            fontdict=self.fontdict,
            labelpad=5.0,
            loc="center"
        )
        # hide the top and right spines of the graph to give a look of Gantt chart
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            sec_ax.spines[spine].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(self.output_path_dir / 'gantt_chart.png', dpi=GlobalConst.DEFAULT_SAVEDPI)
        plt.savefig(self.output_path_dir / 'gantt_chart.svg', format='svg')
        plt.savefig(self.output_path_dir / 'gantt_chart.pdf', format='pdf')
        plt.close()

        return self
    
    @time_decorator
    def build_wbs(
        self,
        draw_groups: bool = False
    ):
        """
        Build a Work Breakdown Structure (WBS) diagram from the WBS dataframe.
        Args:
            draw_groups (bool): Whether to color groups based on their first-order parent.
        """
        # ------- create graph ------- #
        G = nx.DiGraph()
        G.add_edges_from(
            self.wbs_df[
                [
                    WBSKeys.PARENT.value, 
                    WBSKeys.CHILDREN.value
                ]
            ].values
        )

        # ------- layout graph ------- #
        # https://graphviz.org/doc/info/attrs.html
        G.graph["graph"] = {
            # Left-to-right layout
            'rankdir': 'LR',   
            # Vertical spacing between ranks
            'ranksep': '10.0',  
            'splines': 'ortho',
        }

        pos: dict = nx.nx_pydot.graphviz_layout(G, prog="dot")
        n_nodes = len(G.nodes)
        pos_min_x = min(x for x, y in pos.values())
        pos_min_y = min(y for x, y in pos.values())
        pos_max_x = max(x for x, y in pos.values())
        pos_max_y = max(y for x, y in pos.values())
        scale_factor = 1.2
        x_range = [pos_min_x * scale_factor, pos_max_x * scale_factor]
        y_range = [pos_min_y * scale_factor, pos_max_y * scale_factor]

        width = max(20, n_nodes * 0.1)
        height = max(20, n_nodes * 0.1)

        # ------- extract edges ------- #
        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # ------- extract nodes ------- #
        node_x, node_y, text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)

        # ------- create figure ------- #
        fig = plt.figure(figsize=(width, height),facecolor='lightgrey')
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle(
            t=self.title + "\n" + GlobalConst.DEFAULT_WBS_TITLE if self.title else GlobalConst.DEFAULT_WBS_TITLE,
            fontsize=self.fontdict.get("fontsize", 16), 
            fontweight=self.fontdict.get("fontweight", 'bold'),
            family=self.fontdict.get("fontfamily", 'monospace'), 
            y=0.95
        )
        # draw node markers but keep them invisible
        # they are required for layout for the labels and edges
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color='none',
            node_shape='s',
            node_size=3000,
            ax=ax,
        )
        # draws labels as text with bbox so we can compute their extents
        label_artists = {}
        for node in G.nodes():
            x, y = pos[node]
            txt = ax.text(
                x,
                y,
                str(node),
                bbox=dict(boxstyle="round", pad=0.5, fc="white", ec="black", lw=1),
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=self.fontdict.get("fontsize", 10),
                fontfamily=self.fontdict.get("fontfamily", "monospace"),
                fontweight=self.fontdict.get("fontweight", "bold"),
                zorder=3,
            )
            label_artists[node] = txt

        # we need the renderer to compute text extents
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # compute label bbox extents in data coordinates (both x and y)
        label_bboxes = {}
        for node, txt in label_artists.items():
            bbox_disp = txt.get_window_extent(renderer=renderer)
            # convert display (pixel) coords to data coords
            inv = ax.transData.inverted()
            left_bottom = inv.transform((bbox_disp.x0, bbox_disp.y0))
            right_top = inv.transform((bbox_disp.x1, bbox_disp.y1))
            left_x = left_bottom[0]
            right_x = right_top[0]
            bottom_y = left_bottom[1]
            top_y = right_top[1]
            label_bboxes[node] = (left_x, right_x, bottom_y, top_y)

        # draw group background bands for nodes that share the same first-order parent
        # build parent map (child -> parent) from the flattened WBS dataframe
        parent_map = dict(zip(self.wbs_df[WBSKeys.CHILDREN.value], self.wbs_df[WBSKeys.PARENT.value]))

        def get_first_order_parent(node):
            # walk up until parent is 'wbs' (root) or missing; return the first-level parent
            p = parent_map.get(node)
            if p is None or p == 'wbs':
                # node is a top-level group or orphan
                return node if p == 'wbs' else p
            # climb until parent of p is 'wbs'
            while True:
                grand = parent_map.get(p)
                if grand is None or grand == 'wbs':
                    return p
                p = grand

        groups = defaultdict(list)
        for node in G.nodes():
            if node not in label_bboxes:
                continue
            first_parent = get_first_order_parent(node)
            groups[first_parent].append(node)

        # choose colors for groups
        group_keys = [k for k in groups.keys() if k is not None]
        group_color_map = self._create_group_color_map(draw_groups)
        group_alpha = 0.2

        for group_name, nodes_in_group in groups.items():
            # compute vertical span using label bbox y extents
            bottoms = [label_bboxes[n][2] for n in nodes_in_group]
            tops = [label_bboxes[n][3] for n in nodes_in_group]
            if not bottoms or not tops:
                continue
            bottom = min(bottoms) - 0.3
            top = max(tops) + 0.3
            color = group_color_map.get(group_name, (0.9, 0.9, 0.9))
            ax.axhspan(
                bottom, 
                top, 
                facecolor=color, 
                alpha=group_alpha, 
                zorder=0
            )

        # draw edges manually so they attach to label box sides
        for u, v in G.edges():
            if u not in label_bboxes or v not in label_bboxes:
                continue
            x0 = label_bboxes[u][1]  # right side of parent label
            y0 = pos[u][1]
            x1 = label_bboxes[v][0]  # left side of child label
            y1 = pos[v][1]

            # create an arrow that connects right edge of parent to left edge of child
            arr = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle='-|>',
                mutation_scale=20,
                color='black',
                linewidth=2,
                connectionstyle='arc3,rad=0.0',
                zorder=1,
            )
            ax.add_patch(arr)

        # adds a horizontal span accross groups with same first order parent
        ax.set_facecolor('white')
        ax.margins(x=0.4, y=0.2)

        plt.tight_layout()
        plt.savefig(self.output_path_dir / 'work_breakdown_structure.png')
        plt.savefig(self.output_path_dir / 'work_breakdown_structure.svg', format='svg')
        plt.savefig(self.output_path_dir / 'work_breakdown_structure.pdf', format='pdf')
        plt.close()

        return self

        

