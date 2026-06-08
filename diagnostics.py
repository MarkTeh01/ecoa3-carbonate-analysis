import pandas as pd
import statsmodels.api as sm

# Toggle dataset here:
#file_path = "CaCO3_Arag.csv"
#df = pd.read_csv(file_path)
#X = df["%-CaCO3"]
#y = df["%-Arag"]
file_path = "CaCO3_Calc.csv"
df = pd.read_csv(file_path)
X = df["%-CaCO3"]
y = df["%-Calc"]

X_const = sm.add_constant(X)

# Ordinary Least Squares:
ols_model = sm.OLS(y, X_const).fit()
print(ols_model.summary())
# Huber-M Estimation:
rlm_model = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
print(rlm_model.summary())

# Data Diagnostics:
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
student_resid = influence.resid_studentized_external
df["CooksD"] = cooks_d
df["Leverage"] = leverage
df["Studentized_Resid"] = student_resid
n = len(df)
threshold = 4 / n
print("\n===== INFLUENCE DIAGNOSTICS =====")
print(f"Cook's D threshold (4/n): {threshold:.4f}")
print(df[["CooksD", "Leverage", "Studentized_Resid"]].sort_values("CooksD", ascending=False))