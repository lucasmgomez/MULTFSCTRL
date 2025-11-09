import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

task_name = "dmsloc"
subjects = ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06"]

# Set the path where your task CSV files are stored
data_dir = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/results/{task_name}_accuracy_results.csv") 
df = pd.read_csv(data_dir)

## Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="subject", y="accuracy", order=subjects)

# Customize
plt.title(f"{task_name} Accuracy Distribution per Subject")
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.xlabel("Subject")
plt.tight_layout()

# Save figure
save_path = Path(f"/project/def-pbellec/xuan/fmri_dataset_project/results/performance/{task_name}_accuracy_boxplot.png")
os.makedirs(save_path.parent, exist_ok=True)
plt.savefig(save_path, dpi=300)

print(f"Figure saved to: {save_path}")