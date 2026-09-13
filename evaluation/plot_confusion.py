import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

def plot_confusion_matrix():
    eval_json = REPORTS_DIR / "evaluation.json"
    if not eval_json.exists():
        print("No evaluation.json found. Run evaluation.run first.")
        return

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["main_system"]["intent_metrics"]["confusion_matrix"])
    labels = data["main_system"]["intent_metrics"]["labels"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="Intent Classification Confusion Matrix",
        ylabel="True Intent Label",
        xlabel="Predicted Intent Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_file = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to {out_file}")

if __name__ == "__main__":
    plot_confusion_matrix()
