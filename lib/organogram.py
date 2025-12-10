
import json
import textwrap
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt

from pathlib import Path
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch

from .global_const import *
from .util_functions import *

class OrganogramBuilder:

    def __init__(
        self,
        input_path: Path,
        output_path_dir: Path,
        font_dict: dict = None,
        color_map = None,
    ):
        self.input_path = input_path
        self.output_path_dir = output_path_dir

        self.font_dict = font_dict if font_dict else GlobalConst.FONT_DICT
        self.color_map = color_map if color_map else GlobalConst.COLOR_MAP_ROLES

        self.organogram_df = pd.DataFrame()
        self.title = None

        if self.input_path:
            if self.input_path.suffix == '.json':
                self._from_json(self.input_path)
                # YAML is also a possibility in the future or other formats that support hierarchical data (like XML)
            else:
                raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        
        if self.organogram_df is None or self.organogram_df.empty:
            raise ValueError("No data available to build charts.")


    def _clean_data(self):
        pass

    @time_decorator
    def _from_xml(self, file_path: Path):
        raise NotImplementedError("XML input not yet implemented.")

    @time_decorator
    def _from_yaml(self, file_path: Path):
        raise NotImplementedError("YAML input not yet implemented.")

    @time_decorator
    def _from_json(self, file_path: Path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        data = self._flatten_organogram(data)
        self.organogram_df = pd.DataFrame(data)
        self.organogram_df.to_csv(file_path.parent / f'organogram_data_{file_path.stem}.csv', index=False)
        self.organogram_df.info()

    def _create_role_color_map(self, draw_roles: bool = True):
        if not draw_roles:
            return {}
        
        unique_roles = self.organogram_df[OrganogramKeys.ROLE.value].unique()
        role_color_map = {}
        for i, role in enumerate(unique_roles):
            role_color_map[role] = self.color_map(i % self.color_map.N)
        return role_color_map


    def _flatten_organogram(self, data: dict, parent=None,level=0, rows=None):
        """
        Flattens a hierarchical organogram structure into a list of roles with hierarchy info.
        Args:
            data (dict): The organogram data passed recursively for each node.
            parent (str): The parent role name.
            level (int): Current level in the hierarchy.
            row (list): Accumulator for flattened roles.
        """
        if level == 0:
            rows = []    
            self.title = data.get(ProjectKeys.TITLE.value, None)
            root_organogram = data.get(ProjectKeys.ORGANOGRAM.value, None)

            if root_organogram is None:
                raise ValueError(f"No root key {ProjectKeys.ORGANOGRAM.value} found in the input.")
            
            data = root_organogram

        for key, value in data.items():
            if key in IGNORED_KEYS:
                continue

            # is a supervisor with subordinates
            if isinstance(value, dict) and OrganogramKeys.SUPERVISOR_OF.value in value:
                print(f"{'  '*level}Processing supervisor: {value.get(OrganogramKeys.NAME.value, 'Unknown')} at level {level}")
                rows.append({
                    OrganogramKeys.NAME.value: key,
                    OrganogramKeys.REPORTS_TO.value: parent if parent else None,
                    OrganogramKeys.ROLE.value: value.get(OrganogramKeys.ROLE.value, None)
                })
                

                new_data = value.get(OrganogramKeys.SUPERVISOR_OF.value)
                self._flatten_organogram(
                    data=new_data,
                    parent=key,
                    level=level+1,
                    rows=rows
                )
                
            # is data of a person/role
            elif isinstance(value, dict) and OrganogramKeys.SUPERVISOR_OF.value not in value:
                print(f"{'  '*level}Processing role: {key} at level {level}")
                rows.append({
                    OrganogramKeys.NAME.value: key,
                    OrganogramKeys.REPORTS_TO.value: parent if parent else None,
                    OrganogramKeys.ROLE.value: value.get(OrganogramKeys.ROLE.value, None)
                })
            

        return rows


    def build_organogram(
        self,
        label_max_width: int = 20,
        draw_roles: bool = True,
    ):

        fig = plt.figure(figsize=(10, 10))        
        plt.title(self.title + "\n" + GlobalConst.DEFAULT_ORGANOGRAM_TITLE if self.title else GlobalConst.DEFAULT_ORGANOGRAM_TITLE, fontdict=self.font_dict)
        ax = plt.gca()
        ax.axis('off')

        # with matplotlib, we can create a tree structure manually
        # the chart is a top to bottom hierarchy with arrows indicating reporting lines
        # people with same supervisor are aligned horizontally people with same role are grouped together
        # in 1 bbox.
        # type of tree orthogonal which means straight lines with right angle turns
        # we can use FancyArrowPatch to draw arrows between boxes
        pos = {}
        organogram = self.organogram_df.groupby(
            by=[
                OrganogramKeys.REPORTS_TO.value,
                OrganogramKeys.ROLE.value
            ]
        )
        color_role_map = self._create_role_color_map(draw_roles=draw_roles)

        # NOTE: coordinates in matplotlib go from (0,0) at bottom-left to (1,1) at top-right
        def set_pos(name, x, y, level_gap=1/5, sibling_gap=1/5):
            pos[name] = (x, y)
            children = self.organogram_df[self.organogram_df[OrganogramKeys.REPORTS_TO.value] == name][OrganogramKeys.NAME.value].tolist()
            num_children = len(children)
            if num_children > 0:
                start_x = x - (sibling_gap * (num_children - 1)) / 2
                for i, child in enumerate(children):
                    set_pos(child, start_x + i * sibling_gap, y - level_gap, level_gap, sibling_gap)

        # Start positioning from the top-level roles (those without a supervisor)
        top_level_roles = self.organogram_df[self.organogram_df[OrganogramKeys.REPORTS_TO.value].isnull()][OrganogramKeys.NAME.value].tolist()
        gap_title_top = 0.1
        for i, role in enumerate(top_level_roles):
            set_pos(role, 1 / 2, 1 - gap_title_top)

        for idx, row in self.organogram_df.iterrows():
            name = row[OrganogramKeys.NAME.value]
            role = row[OrganogramKeys.ROLE.value]
            x, y = pos[name]
            text = plt.text(
                x, 
                y, 
                textwrap.fill(f"{name}\n{role}", width=label_max_width),
                ha="center", 
                va="center", 
                fontdict=self.font_dict,
                bbox=dict(
                    boxstyle="round,pad=0.3", 
                    fc=color_role_map.get(role, self.color_map(idx % self.color_map.N)) if draw_roles else 'lightgray', 
                    ec="black", 
                    lw=2
                )
            )
            # computes bounding box during the drawing phase
            # to get bbox dimensions. these dimensions are in display coords
            # so we need to transform them back to data coords if needed
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = text.get_bbox_patch()
            w = bbox.get_width() / fig.dpi / fig.get_size_inches()[0]
            h = bbox.get_height() / fig.dpi / fig.get_size_inches()[1]
            print(f"Role: {name}, Position: ({x:.2f}, {y:.2f}), BBox width: {w:.2f}, height: {h:.2f}")

            supervisor = row[OrganogramKeys.REPORTS_TO.value]
            if pd.notna(supervisor):
                sx, sy = pos[supervisor]
                arrow = FancyArrowPatch(
                    # the arrow starts at bottom center of supervisor box to top center of subordinate box
                    (sx, sy - h / 2), (x, y + h / 2),
                    arrowstyle='-', 
                    mutation_scale=10, 
                    color='gray', 
                    lw=1.5
                )
                ax.add_patch(arrow)


        plt.tight_layout()
        plt.savefig(self.output_path_dir / 'organogram_chart.png', dpi=GlobalConst.DEFAULT_SAVEDPI)
        plt.savefig(self.output_path_dir / 'organogram_chart.svg', format='svg')
        plt.savefig(self.output_path_dir / 'organogram_chart.pdf', format='pdf')
        plt.close()

        return self


