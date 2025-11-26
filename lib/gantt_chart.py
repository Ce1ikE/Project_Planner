import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

DATE_FORMAT = '%Y-%m-%d'
TITLE_SIZE = 12
TITLE_FONT_WEIGHT = "bold"
FONT_COLOR = "#6C6C6C"
X_LABEL = "Timeline"
Y_LABEL = "Tasks"
LABEL_SIZE = 8
DAY_FONT_SIZE = 8
DAY_FONT_WEIGHT = "bold"
DAY_FONT_COLOR = FONT_COLOR
MONTH_FONT_SIZE = 10
MONTH_FONT_WEIGHT = "bold"
MONTH_FONT_COLOR = FONT_COLOR
DEPENDENCIES_COLOR = "#767170"

class GanttChart:
    def __init__(
        self,
        input_path: Path = None,
        output_path_dir: Path = None,
        data_date_format: str = DATE_FORMAT,
        title_size: int = TITLE_SIZE,
        title_font_weight: int =  TITLE_FONT_WEIGHT,
        y_label = Y_LABEL,
        x_label = X_LABEL,
        day_font_size = DAY_FONT_SIZE,
        day_font_weight = DAY_FONT_WEIGHT,
        day_font_color = DAY_FONT_COLOR,
        month_font_size = MONTH_FONT_SIZE,
        month_font_weight = MONTH_FONT_WEIGHT,
        month_font_color = MONTH_FONT_COLOR,
        dependencies_color = DEPENDENCIES_COLOR,
    ):
        self.input_path = input_path
        self.output_path_dir = output_path_dir
        self.output_path_dir.mkdir(parents=True, exist_ok=True)
        
        self.y_label = y_label
        self.x_label = x_label
        
        self.day_font_size = day_font_size
        self.day_font_weight = day_font_weight
        self.day_font_color = day_font_color

        self.month_font_size = month_font_size
        self.month_font_weight = month_font_weight
        self.month_font_color = month_font_color

        self.dependencies_color = dependencies_color
        
        self.fontdict = {
            "fontfamily": "monospace",
            "fontsize": 10,
            "fontweight": "bold",
            "color": FONT_COLOR,
        }

        self.title_size = title_size
        self.title_font_weight = title_font_weight
        self.data_date_format = data_date_format
        
        self.tasks_df: pd.DataFrame = None
        self.fig = None

        if self.input_path:
            if self.input_path.suffix == '.csv':
                self._from_csv(self.input_path)
            elif self.input_path.suffix == '.json':
                self._from_json(self.input_path)
                # YAML is also a possibility in the future
            else:
                raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        
        if self.tasks_df is None or self.tasks_df.empty:
            return
        
        self.tasks_df['start_date'] = pd.to_datetime(self.tasks_df['start_date'], format=self.data_date_format)
        self.tasks_df['end_date'] = pd.to_datetime(self.tasks_df['end_date'], format=self.data_date_format)
        self.tasks_df['duration'] = (self.tasks_df['end_date'] - self.tasks_df['start_date']).dt.days
        self.tasks_df.fillna({'resources': {}, 'costs': 0}, inplace=True)

        self.tasks_df.info()

    def _from_csv(self, file_path: Path):
        self.tasks_df = pd.read_csv(file_path)

    def _from_json(self, file_path: Path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        data = self._flatten_wbs(data)
        self.tasks_df = pd.DataFrame(data)
        self.tasks_df.to_csv(file_path.parent / f'gantt_data_{file_path.stem}.csv', index=False)

    def _flatten_wbs(self, data: dict, parent=None, level=0, rows=None):
        # Flatten the nested `wbs` structure into rows suitable for a Gantt chart.
        # We expect the input JSON to have a top-level `wbs` key. If present,
        # start from `data['wbs']`. Only items that define a start_date/end_date
        # (i.e. schedulable tasks) will be returned.

        if level == 0:
            rows = []
            # preserve the top-level title if present
            self.title = data.get("title", "Gantt Chart Visualization")
            # start from the 'wbs' section when available
            work_items = data.get("wbs", data)
        else:
            work_items = data

        for key, value in work_items.items():
            if isinstance(value, dict):
                # leaf task (has scheduling info)
                if "start_date" in value or "end_date" in value or "duration" in value:
                    rows.append({
                        "task_name": key,
                        "start_date": value.get("start_date", None),
                        "end_date": value.get("end_date", None),
                        "resources": value.get("resources", []),
                        "costs": value.get("costs", 0),
                        "dependencies": value.get("dependencies", []),
                        "parent": parent if parent else None,
                        "child": key,
                    })
                else:
                    # group node: recurse into its children and pass current key as parent
                    self._flatten_wbs(value, key, level + 1, rows)

        return rows

    def _build_week_ticks(self, start_date, end_date):
        mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        return mondays, [d.strftime('%d') for d in mondays]

    def build_gantt_chart(self, draw_dependencies: bool = False, draw_groups: bool = False):

        week_positions, week_labels = self._build_week_ticks(
            self.tasks_df['start_date'].min(), 
            self.tasks_df['end_date'].max()
        )

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_title(self.title, fontdict=self.fontdict)
        ax.tick_params(axis='both')
        # 1 task is a bar with start and end date
        tasks = self.tasks_df.sort_values(by=['start_date', 'task_name'], ascending=False)
        # foreach task we'll create a bar and 
        # add annotations
        # draw bars with automatic color assignment (using a colormap)
        cmap = plt.get_cmap('tab20')
        n_colors = getattr(cmap, 'N', 20)

        # store geometry for optional dependency drawing
        bar_info = {}
        groups = defaultdict(list)
        for idx, task in enumerate(tasks.itertuples(index=False)):
            task_name = task.task_name
            start = task.start_date
            end = task.end_date
            parent = task.parent or "Ungrouped"
            duration = (end - start).days
            resources = list(task.resources) if isinstance(task.resources, (list, tuple)) else task.resources

            color = cmap(idx % n_colors)
            bar = ax.barh(
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
            if len(resources_str) > 36:
                resources_str = resources_str[:33] + '...'
            label = f"{duration}d | €{task.costs}"
            if resources_str:
                label = label + ' | ' + resources_str

            gap = 2.5
            # put text next to the right of the bar
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() + gap, y_center),
                ha='left',
                va='center',
                fontsize=8,
                color='black',
                fontfamily='monospace',
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8)
            )

        # draw horizontal lines on top of the graph to group tasks by parent
        if draw_groups:
            color_groups = plt.get_cmap('tab20')
            group_color_map = {g: color_groups(i % getattr(color_groups, 'N', 20)) for i, g in enumerate(groups.keys())}
            group_alpha = 0.2

            for group_name, infos in groups.items():
                ys = [i['y'] for i in infos]
                bottoms = [i['bottom'] for i in infos]
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
                    alpha=group_alpha, 
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
                        
                        if pred.get("dependencies_marker", 0) is not 0 or succ.get("dependencies_marker", 0) is not 0:
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
            label.set_fontfamily("monospace")

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
        plt.savefig(self.output_path_dir / 'gantt_chart.png', dpi=300)
        plt.savefig(self.output_path_dir / 'gantt_chart.svg', format='svg')
        plt.savefig(self.output_path_dir / 'gantt_chart.pdf', format='pdf')
        plt.close()
