import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Parameters
# =========================
INPUT_FILE = "Q1_19_Seq_SHAP_Analysis.xlsx"
TARGET_COL = "Exp_Log2_MIC"

OUTDIR = Path("Differential_Analysis_Output")
OUTDIR.mkdir(exist_ok=True)

OUT_SUMMARY = OUTDIR / "Differential_Feature_Summary_Log2MIC.xlsx"
OUT_ZDATA = OUTDIR / "Differential_Analysis_Zscore_Data.xlsx"
OUT_TOPDATA = OUTDIR / "Top_Feature_Raw_Data.xlsx"
OUT_PNG = OUTDIR / "Figure_HighSpearman_SignStable.png"
OUT_PDF = OUTDIR / "Figure_HighSpearman_SignStable.pdf"

# =========================
# Load data
# =========================
df = pd.read_excel(INPUT_FILE)
assert TARGET_COL in df.columns, "Missing Exp_Log2_MIC column"

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(TARGET_COL)

y = df[TARGET_COL].values
results = []

# =========================
# Core differential analysis
# =========================
for feat in num_cols:
    x = df[feat].values
    if np.nanstd(x) < 1e-8:
        continue

    rho, pval = spearmanr(x, y)
    results.append({
        "Feature": feat,
        "SpearmanR_Log2MIC": rho,
        "P_value": pval,
        "Sign": "Positive" if rho > 0 else "Negative",
        "Abs_SpearmanR": abs(rho)
    })

res_df = pd.DataFrame(results)

# =========================
# Effect strength classification
# =========================
def classify_strength(r):
    if r >= 0.4:
        return "Strong"
    elif r >= 0.2:
        return "Moderate"
    else:
        return "Weak"

res_df["Effect_Strength"] = res_df["Abs_SpearmanR"].apply(classify_strength)

# =========================
# Sort + add Num index
# =========================
res_df = res_df.sort_values("Abs_SpearmanR", ascending=False).reset_index(drop=True)
res_df.insert(0, "Num", np.arange(1, len(res_df) + 1))

# Export summary
res_df.to_excel(OUT_SUMMARY, index=False)

# =========================
# Export Z-score differential matrix
# =========================
z_df = df[num_cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
z_df[TARGET_COL] = df[TARGET_COL]
z_df.to_excel(OUT_ZDATA, index=False)

# =========================
# Export top feature raw data
# =========================
top_feats = res_df[res_df["Effect_Strength"].isin(["Strong", "Moderate"])]["Feature"].tolist()
top_raw_df = df[top_feats + [TARGET_COL]]
top_raw_df.to_excel(OUT_TOPDATA, index=False)

# =========================
# SCI-ready figure (conditional Num labeling)
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

strong = res_df[res_df["Effect_Strength"] == "Strong"]
moderate = res_df[res_df["Effect_Strength"] == "Moderate"]
others = res_df[res_df["Effect_Strength"] == "Weak"]

plt.figure(figsize=(6, 5))

# Scatter all points
plt.scatter(
    others["Num"],
    others["SpearmanR_Log2MIC"],
    alpha=0.5,
    label="Weak"
)

plt.scatter(
    moderate["Num"],
    moderate["SpearmanR_Log2MIC"],
    alpha=0.7,
    label="Moderate"
)

plt.scatter(
    strong["Num"],
    strong["SpearmanR_Log2MIC"],
    s=70,
    label="Strong"
)

# =========================
# Decide which points to annotate
# =========================
if len(strong) > 0:
    to_annotate = strong
else:
    to_annotate = moderate.head(3)

# Annotate selected points with Num
for _, row in to_annotate.iterrows():
    plt.text(
        row["Num"],
        row["SpearmanR_Log2MIC"],
        str(int(row["Num"])),
        fontsize=9,
        ha="center",
        va="bottom"
    )

# Reference lines
plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(0.4, linestyle=":", linewidth=1)
plt.axhline(-0.4, linestyle=":", linewidth=1)

plt.xlabel("Feature rank (by |SpearmanR|)")
plt.ylabel("Spearman correlation with Exp_Log2_MIC")
plt.title("High-confidence determinants of Exp_Log2_MIC")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=600)
plt.savefig(OUT_PDF)
plt.close()

print("✅ Differential analysis completed.")
print(f"📁 All outputs saved in: {OUTDIR.resolve()}")
