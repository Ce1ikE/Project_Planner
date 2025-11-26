import json
from pathlib import Path
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class WBS:
    def __init__(
        self,
        input_path: Path = None,
        output_path_dir: Path = None,
    ):
        self.wbs_df: pd.DataFrame = None
        self.fig = None

        self.title = None

        self.input_path = input_path

        self.output_path_dir = output_path_dir
        self.output_path_dir.mkdir(parents=True, exist_ok=True)

        if self.input_path:
            if self.input_path.suffix == '.json':
                self._from_json(self.input_path)
            elif self.input_path.suffix == '.csv':
                self._from_csv(self.input_path)
                # YAML is also a possibility in the future
            else:
                raise ValueError(f"Unsupported file format: {self.input_path.suffix}") 

        if self.wbs_df is None or self.wbs_df.empty:
            return
        self.wbs_df.info()

    def _from_json(self, data_path: Path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        wbs_data = self._flatten_wbs(data)
        self.wbs_df = pd.DataFrame(wbs_data)
        self.wbs_df.to_csv(data_path.parent / f'wbs_data_{data_path.stem}.csv', index=False)

    def _from_csv(self, data_path: Path):
        self.wbs_df = pd.read_csv(data_path)

    def _flatten_wbs(self, data: dict, parent=None, level=0, edges=None):
        # Build parent->child edges from the nested `wbs` section.
        # Only traverse the hierarchical WBS structure (data['wbs'] when present).
        if level == 0:
            edges = []
            self.title = data.get("title", "WBS Visualization")
            work_items = data.get("wbs", data)
        else:
            work_items = data


        for key, value in work_items.items():
            if not isinstance(value, dict):
                continue
            edges.append({
                "parent": parent if parent else "wbs",
                "child": key
            })

            if isinstance(value, dict):
                # recurse into nested groups/tasks
                self._flatten_wbs(value, key, level + 1, edges)

        return edges

    def build_wbs(self):
        # ------- create graph ------- #
        G = nx.DiGraph()
        G.add_edges_from(self.wbs_df[['parent', 'child']].values)

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

        cmap = plt.get_cmap('tab20')

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
            t="Work Breakdown Structure - " + self.title,
            fontsize=16, 
            fontweight='bold', 
            family='monospace', 
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
                fontsize=16,
                fontfamily='monospace',
                fontweight='bold',
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
        parent_map = dict(zip(self.wbs_df['child'], self.wbs_df['parent']))

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
        group_color_map = {g: cmap(i % getattr(cmap, 'N', 20)) for i, g in enumerate(group_keys)}
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

        

