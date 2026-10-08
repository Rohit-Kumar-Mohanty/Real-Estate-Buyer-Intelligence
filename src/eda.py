from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_eda_figures(df, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ax = df["acquisition_purpose"].value_counts().plot(kind="bar", figsize=(8, 5))
    ax.set_title("Acquisition Purpose")
    ax.set_xlabel("Purpose")
    ax.set_ylabel("Buyers")
    plt.tight_layout()
    plt.savefig(output / "acquisition_purpose.png", dpi=160)
    plt.close()

    ax = df["client_type"].value_counts().plot(kind="bar", figsize=(8, 5))
    ax.set_title("Client Type")
    ax.set_xlabel("Client Type")
    ax.set_ylabel("Buyers")
    plt.tight_layout()
    plt.savefig(output / "client_type.png", dpi=160)
    plt.close()

    ax = df["total_investment"].plot(kind="hist", bins=30, figsize=(8, 5))
    ax.set_title("Total Investment Distribution")
    ax.set_xlabel("Total Investment")
    plt.tight_layout()
    plt.savefig(output / "total_investment_distribution.png", dpi=160)
    plt.close()
