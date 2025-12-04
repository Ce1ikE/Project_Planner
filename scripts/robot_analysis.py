import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from lib.global_const import *
from lib.radar_chart import radar_factory

def run(results_dir: Path, data_dir: Path):

    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(data_dir)
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
    df_robot_arms.to_csv(results_dir / "robot_arms_data.csv", index=False)
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
    df_other.to_csv(results_dir / "other_costs_data.csv", index=False)
    #   Column         Non-Null Count  Dtype
    # ---  ------         --------------  -----
    # 0   Category       5 non-null      object
    # 1   Subcategory    5 non-null      object
    # 2   Item/Model     5 non-null      object
    # 3   Unit_Cost_EUR  5 non-null      object
    print("robot transport:" + "="*20)
    df_robot_transport.drop(columns=["Payload_kg", "Reach_mm"], inplace=True)
    df_robot_transport.info()
    df_robot_transport.to_csv(results_dir / "robot_transport_data.csv", index=False)
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

    df_specs["Payload_mid"] = df_specs["Payload_kg"].apply(midpoint)
    df_specs["Reach_num"] = df_specs["Reach_mm"].apply(to_num)
    df_specs.sort_values(by=["Reach_num"], inplace=True)
    plt.figure()
    for idx, row in df_specs.iterrows():
        plt.scatter(
            row["Payload_mid"], 
            row["Reach_num"], 
            s=100, 
            label=row["Item/Model"],
            marker='X',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
   
    plt.xlabel("Payload (kg)", fontdict=FONT_DICT)
    plt.ylabel("Reach (mm)", fontdict=FONT_DICT)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')

    plt.yticks([500, 1000, 3000, 4000, 5000], 
               ["500 mm", "1000 mm", "3m", "4m", "5m"])
    plt.xticks([1, 10, 100, 1000], 
               ["1 kg", "10 kg", "100 kg", "1000 kg"])
    
    for label in plt.gca().get_xticklabels():
        label.set_fontsize(8)
        label.set_fontfamily('monospace')
        print(label.get_text())

    for label in plt.gca().get_yticklabels():
        label.set_fontsize(8)
        label.set_fontfamily('monospace')
        print(label.get_text())


    plt.title("Robot Arms Specifications", fontdict=FONT_DICT)
    plt.legend(
        df_specs["Item/Model"].unique(), 
        title="Robot Model"
    )
    plt.grid(linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    plt.savefig(results_dir / "robot_arms_specs_scatter.png", format='png', dpi=300)
    plt.savefig(results_dir / "robot_arms_specs_scatter.svg", format='svg')
    plt.close()


    # plt.figure()
    # N = len(df_robot_transport["Item/Model"].where(lambda x: "Export_" in x))
    # categories = df_robot_transport["Item/Model"].tolist()
    # values = df_robot_transport["Unit_Cost_EUR"].tolist()
    
    # plt.barh(
    #     y=categories,
    #     width=values,
    #     color='skyblue',
    #     edgecolor='black',
    # )

    # plt.xlabel("Unit Cost (EUR)", fontdict=FONT_DICT)
    # plt.title("Robot Transport Unit Costs", fontdict=FONT_DICT)
    # plt.tight_layout()
    # plt.savefig(results_dir / "robot_transport_unit_costs_barh.png", format='png', dpi=300)
    # plt.savefig(results_dir / "robot_transport_unit_costs_barh.svg", format='svg')
    # plt.close()

 