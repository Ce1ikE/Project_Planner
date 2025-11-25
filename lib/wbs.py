import json
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
        self.wbs_df.to_csv(data_path.parent / f'{data_path.stem}.csv', index=False)

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
                "parent": parent if parent else "root",
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
            'rankdir': 'LR',   # Left-to-right layout
            'ranksep': '150.0',  # Vertical spacing between ranks
            'nodesep': '100.0',   # Horizontal spacing between nodes
            'pad': '0.5',
            'splines': 'ortho',
            'overlap': 'false',
            'fontsize': '10',
            'fontname': 'monospace',
        }
        # G.graph["node"] = {
        #     'shape': 'box',
        #     'style': 'filled',
        #     'width': '2',
        #     'height': '1',
        #     'fixedsize': 'true',
        #     'color': 'darkblue',
        #     'fillcolor': 'lightblue',
        #     'fontname': 'monospace',
        #     'fontsize': '10',
        # }
        # G.graph["edge"] = {
        #     'fontname': 'monospace',
        #     'fontsize': '8',
        # }

        pos: dict = nx.nx_pydot.graphviz_layout(G, prog="dot")

        n_nodes = len(G.nodes)
        
        pos_min_x = min(x for x, y in pos.values())
        pos_min_y = min(y for x, y in pos.values())
        pos_max_x = max(x for x, y in pos.values())
        pos_max_y = max(y for x, y in pos.values())
        scale_factor = 1.2
        x_range = [pos_min_x * scale_factor, pos_max_x * scale_factor]
        y_range = [pos_min_y * scale_factor, pos_max_y * scale_factor]

        width = max(20, n_nodes * 0.2)
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

        # ------- create figure (networkx) ------- #
        fig = plt.figure(figsize=(width, height))

        fig.suptitle(
            t="Work Breakdown Structure - " + self.title,
            fontsize=16, fontweight='bold', family='monospace',y=0.95
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color='none',
            node_shape='s',
            node_size=2000,
        )
        # draws the graph's labels
        nx.draw_networkx_labels(
            G,
            pos,
            bbox=dict(boxstyle="round", pad=0.5, fc="white", ec="black", lw=1),
            horizontalalignment='left',
            verticalalignment='center',
            font_size=8,
            font_family='monospace',
            font_weight='bold',
        )
        # draws the graph's edges
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle='-|>',
            arrowsize=20,
            edge_color='black',
            style='dashed',
            width=2,
            min_source_margin=15,
            min_target_margin=15,
        )
        fig.set_facecolor('lightgray')
        plt.gca().set_facecolor('lightgray')
        plt.gca().axes.margins(x=0.5,y=0.0)
        plt.tight_layout()
        plt.savefig(self.output_path_dir / 'wbs_visualization_networkx.png')
        plt.savefig(self.output_path_dir / 'wbs_visualization_networkx.svg', format='svg')
        plt.savefig(self.output_path_dir / 'wbs_visualization_networkx.pdf', format='pdf')
        plt.close()

        

