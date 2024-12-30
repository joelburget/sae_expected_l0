import pandas as pd
import matplotlib.pyplot as plt

file_path = "baselines.csv"
numeric_columns = ["L0", "MSE", "CE"]

if __name__ == "__main__":
    data = pd.read_csv(file_path)
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    methods = data["Method"].unique()
    plot_data = {method: data[data["Method"] == method] for method in methods}

    for metric in ["MSE", "CE"]:
        plt.figure(figsize=(10, 6))
        for method, df in plot_data.items():
            plt.plot(df["L0"], df[metric], label=method)
        plt.xlabel("L0")
        plt.ylabel(metric)
        plt.title(f"L0 vs {metric}")
        plt.legend()
        plt.grid(True)

    plt.show()
