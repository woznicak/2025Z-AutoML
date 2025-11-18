import pandas as pd

df = pd.read_csv("results\\usage_100\\tunability_analysis.csv")
#df = pd.read_csv("results\\xgboost\\usage_100\\tunability_analysis.csv")

df = df[["model", "sampling_method", "dataset_id", "tunability_risk_diff"]]
df["model__dataset"] = df["model"] + "__" + df["dataset_id"].astype(str)
df = df[["sampling_method", "model__dataset", "tunability_risk_diff"]]

df.to_csv("results/cdd/cdd.csv", index=False)
#df.to_csv("results/cdd/cdd_xgboost.csv", index=False)