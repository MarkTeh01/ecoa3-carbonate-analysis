import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import t

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

rlm_model = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
x_vals = np.linspace(X.min(), X.max(), 200)
x_vals_const = sm.add_constant(x_vals)
y_pred = rlm_model.predict(x_vals_const)
cov = rlm_model.cov_params()
se_mean = np.sqrt(np.sum((x_vals_const @ cov) * x_vals_const, axis=1))
df_resid = len(y) - len(rlm_model.params)
t_crit = t.ppf(0.975, df_resid)
ci_upper = y_pred + t_crit * se_mean
ci_lower = y_pred - t_crit * se_mean
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="black", s=60)
plt.plot(x_vals, y_pred, linewidth=2, color="black")
plt.fill_between(x_vals, ci_lower, ci_upper, alpha=0.3, label="95% CI")
plt.xlabel("% CaCO3")
#plt.ylabel("% Aragonite")
plt.ylabel("% Calcite")
plt.tight_layout()
plt.show()