import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from lib.global_const import *

def run(results_dir: Path, data_file: Path):

    np.random.seed(GlobalConst.RANDOM_SEED)

    df = pd.read_csv(data_file)
    # seperate data into relevant sections based on 'Category' and 'Subcategory'
    df.info()
    #   Column         Non-Null Count  Dtype
    # ---  ------         --------------  -----
    # 0   Category       34 non-null     object
    # 1   Subcategory    34 non-null     object
    # 2   Item/Model     34 non-null     object
    # 3   Unit_Cost_EUR  26 non-null     float64
    # 4   Payload_kg     8 non-null      object
    # 5   Reach_mm       23 non-null     object
    
    df_auto = df[df["Category"] == "Automation_Cost_Estimate"]
    df_robot_arms = df[df["Category"] == "Robot_Reference_Specifications"]
    df_other = df_auto[df_auto["Subcategory"] == "Other_Costs"]
    df_robot_transport = df_auto[df_auto["Subcategory"] != "Other_Costs"]

    print("robot arms:" + "="*20)
    df_robot_arms.info()
    df_robot_arms.to_csv(data_file.parent / f"robot_arms_data_{data_file.stem}.csv", index=False)
    #   Column         Non-Null Count  Dtype
    # ---  ------         --------------  -----
    # 0   Category       8 non-null      object
    # 1   Subcategory    8 non-null      object
    # 2   Item/Model     8 non-null      object
    # 3   Unit_Cost_EUR  8 non-null      object
    # 4   Payload_kg     8 non-null      object
    # 5   Reach_mm       8 non-null      object
    print("other costs:" + "="*20)
    df_other.drop(columns=["Payload_kg", "Reach_mm"], inplace=True)
    df_other.info()
    df_other.to_csv(data_file.parent / f"other_costs_data_{data_file.stem}.csv", index=False)
    #   Column         Non-Null Count  Dtype
    # ---  ------         --------------  -----
    # 0   Category       5 non-null      object
    # 1   Subcategory    5 non-null      object
    # 2   Item/Model     5 non-null      object
    # 3   Unit_Cost_EUR  5 non-null      object
    print("robot transport:" + "="*20)
    df_robot_transport.drop(columns=["Payload_kg", "Reach_mm"], inplace=True)
    df_robot_transport.info()
    df_robot_transport.to_csv(data_file.parent / f"robot_transport_data_{data_file.stem}.csv", index=False)
    #   Column         Non-Null Count  Dtype
    # ---  ------         --------------  -----
    # 0   Category       21 non-null     object
    # 1   Subcategory    21 non-null     object
    # 2   Item/Model     21 non-null     object
    # 3   Unit_Cost_EUR  21 non-null     object

    # Chart 5: Robot specifications scatter
    df_specs = df_robot_arms.copy()

    def midpoint(x):
        if isinstance(x, str) and "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return None

    def to_num(x):
        if isinstance(x, str) and "up to" in x:
            return float(x.replace("up to", "").strip())
        return None
    
    def to_min_num(x):
        if isinstance(x, str) and "-" in x:
            a, b = x.split("-")
            return float(a)
        return None

    def to_max_num(x):
        if isinstance(x, str) and "-" in x:
            a, b = x.split("-")
            return float(b)
        return None
    
    cmap = plt.get_cmap("tab10")

    df_specs["Payload_mid"] = df_specs["Payload_kg"].apply(midpoint)
    df_specs["Payload_min"] = df_specs["Payload_kg"].apply(to_min_num)
    df_specs["Payload_max"] = df_specs["Payload_kg"].apply(to_max_num)
    df_specs["Reach_num"] = df_specs["Reach_mm"].apply(to_num)
    df_specs.sort_values(by=["Reach_num"], inplace=True)

    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)

    y_start = 0.0
    for idx, row in df_specs.iterrows():
        color = cmap(idx % cmap.N)
        payload_mid = row["Payload_mid"]
        reach = row["Reach_num"]
        payload_min = row["Payload_min"]
        payload_max = row["Payload_max"]

        ax.scatter(
            payload_mid, 
            reach, 
            s=100, 
            label=row["Item/Model"],
            marker='X',
            alpha=0.7,
            edgecolors='black',
            color=color,
            linewidths=0.5,
        )

        # Assuming the span starts at a very low y-value, e.g., 1 (or the lowest tick on the log scale)
        # We need to calculate the width and center x-position for ax.bar
        bar_width = payload_max - payload_min
        
        # Calculate the starting y-point. Since you are using a log scale, 
        # starting at the lowest visual limit (e.g., 1) is a good approximation
        # Let's start the bar slightly above the x-axis to be visible on log scale
        bar_height = reach - y_start
        ax.bar(
            x=payload_mid,         # Center of the bar
            height=bar_height,     # Height of the bar (up to reach)
            width=bar_width,       # Width of the bar (payload range)
            bottom=y_start,        # Starting y-position
            color=color,
            alpha=0.2,
            edgecolor=color,
            linewidth=0.5,
            align='center',        # Bar is centered on payload_mid
            zorder=0,              # Ensure it's behind the scatter point
        )
   
    ax.set_ylim(10, 5000)
    plt.xlabel("Payload (kg)", fontdict=GlobalConst.FONT_DICT)
    plt.ylabel("Reach (mm)", fontdict=GlobalConst.FONT_DICT)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.yticks([10, 100, 500, 1000, 3000, 4000, 5000], 
               ["10 mm","100 mm","500 mm", "1000 mm", "3m", "4m", "5m"])
    plt.xticks([1, 10, 100, 1000], 
               ["1 kg", "10 kg", "100 kg", "1000 kg"])
    
    for label in ax.get_xticklabels():
        label.set_fontsize(GlobalConst.FONT_DICT.get("fontsize", 10))
        label.set_fontfamily(GlobalConst.FONT_DICT.get("fontfamily", "monospace"))
        print(label.get_text())

    for label in ax.get_yticklabels():
        label.set_fontsize(GlobalConst.FONT_DICT.get("fontsize", 10))
        label.set_fontfamily(GlobalConst.FONT_DICT.get("fontfamily", "monospace"))
        print(label.get_text())


    plt.title("Robot Arms Specifications", fontdict=GlobalConst.FONT_DICT)
    L = ax.legend(
        title="Robot Model",
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        ncol=1,
        fontsize=8,
        shadow=True,
        framealpha=0.95,
    )
    plt.setp(
        L.texts, 
        fontfamily=GlobalConst.FONT_DICT.get("fontfamily", "monospace"),
        color=GlobalConst.FONT_DICT.get("color", "#6C6C6C")
    )
    plt.setp(
        L.get_title(),
        fontfamily=GlobalConst.FONT_DICT.get("fontfamily", "monospace"),
        fontsize=GlobalConst.FONT_DICT.get("fontsize", 10),
        fontweight=GlobalConst.FONT_DICT.get("fontweight", "bold"),
        color=GlobalConst.FONT_DICT.get("color", "#6C6C6C")
    )
    plt.grid(linestyle='--', alpha=0.7, which='both')
    # Use subplots_adjust to give room for legend without squeezing chart
    plt.subplots_adjust(right=0.75)
    plt.savefig(results_dir / "robot_arms_specs_scatter.png", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "robot_arms_specs_scatter.svg", format='svg', bbox_inches='tight')
    plt.savefig(results_dir / "robot_arms_specs_scatter.pdf", format='pdf', bbox_inches='tight')
    plt.close()


    df_robot_transport_products = df_robot_transport[df_robot_transport["Subcategory"].str.contains("Products_")]
    df_robot_transport_products = df_robot_transport_products[df_robot_transport_products["Item/Model"].str.contains(r"Import_[a-zA-Z]*|Export_[a-zA-Z]*", regex=True)] 
    df_robot_transport_products["Item/Model"] = df_robot_transport_products["Item/Model"].str.replace(r"Import_|Export_",lambda x: "", regex=True)
    print(df_robot_transport_products)
    
    df_robot_transport_total = df_robot_transport[df_robot_transport["Subcategory"].str.contains("Total_")]
    print(df_robot_transport_total)    

    categories_products = df_robot_transport_products["Item/Model"].tolist()
    values_products = df_robot_transport_products["Unit_Cost_EUR"].tolist()

    plt.figure()
    plt.barh(
        y=categories_products,
        width=values_products,
        color='skyblue',
        edgecolor='black',
    )

    plt.xlabel("Unit Cost (EUR)", fontdict=GlobalConst.FONT_DICT)
    plt.title("Robot Transport Unit Costs", fontdict=GlobalConst.FONT_DICT)
    plt.tight_layout()
    plt.savefig(results_dir / "robot_transport_unit_costs_barh.png", format='png', dpi=300)
    plt.savefig(results_dir / "robot_transport_unit_costs_barh.svg", format='svg')
    plt.close()

    df_robot_transport_total_additional = df_robot_transport_total[df_robot_transport_total["Item/Model"].str.contains("Base") == False]
    df_robot_transport_total_base = df_robot_transport_total[df_robot_transport_total["Item/Model"].str.contains("Base")]

    categories_total_add = df_robot_transport_total_additional["Item/Model"].tolist()
    values_total_add = df_robot_transport_total_additional["Unit_Cost_EUR"].tolist()

    categories_total_base = df_robot_transport_total_base["Item/Model"].tolist()
    categories_total_base = [cat.replace("(Base)", "") for cat in categories_total_base]
    values_total_base = df_robot_transport_total_base["Unit_Cost_EUR"].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # Plot Base Cost and then overlay Additional Costs
    ax.bar(
        x=categories_total_base,
        height=values_total_add,
        color='red',
        alpha=0.8,
        edgecolor='black',
    )
    ax.bar(
        x=categories_total_base,
        height=values_total_base,
        color='lightgreen',
        edgecolor='black',
    )
    plt.legend(
        labels=["Additional Costs", "Base Cost"],
        loc='upper right',
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(1, 1.05),
    )
    for label in ax.get_xticklabels():
        label.set_fontsize(GlobalConst.FONT_DICT.get("fontsize", 10))
        label.set_fontfamily(GlobalConst.FONT_DICT.get("fontfamily", "monospace"))
        label.set_rotation(15)
        print(label.get_text())

    for label in ax.get_yticklabels():
        label.set_fontsize(GlobalConst.FONT_DICT.get("fontsize", 10))
        label.set_fontfamily(GlobalConst.FONT_DICT.get("fontfamily", "monospace"))
        print(label.get_text())

    plt.grid(linestyle='--', alpha=0.7, which='major', axis='y')
    plt.ylabel("Unit Cost (EUR)", fontdict=GlobalConst.FONT_DICT)
    plt.xlabel("Robot Transport Options", fontdict=GlobalConst.FONT_DICT)
    plt.title("Robot Transport Total Costs", fontdict=GlobalConst.FONT_DICT)
    plt.tight_layout()
    plt.savefig(results_dir / "robot_transport_total_costs_barh.png", format='png', dpi=300)
    plt.savefig(results_dir / "robot_transport_total_costs_barh.svg", format='svg')
    plt.close()

 