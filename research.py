import pandas as pd
import statsmodels.api as sm

df = pd.read_excel('research_project.xlsx', sheet_name="al")

df_no_na = df.copy()

df_no_na.dropna(inplace=True)

df_subset = df.copy()

df_subset = df_subset.iloc[:, 2:]

summary_table = df_subset.describe()
print(summary_table)

medians = df_subset.median()
print(medians)

correlation_matrix = df_subset.corr()
print(correlation_matrix)

# Regression 1: homeless per capita = sustainable per capita
X = df['sus_iss_per_cap']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = df['homeless_per_cap']
model1 = sm.OLS(y, X).fit()
print(model1.summary())

# Regression 2: homeless per capita = social per capita
X = df['soc_iss_per_cap']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = df['homeless_per_cap']
model2 = sm.OLS(y, X).fit()
print(model2.summary())

# Regression 3: % chg homeless = sustainable per capita
X = df_no_na['sus_iss_per_cap']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = df_no_na['pct_chg_homeless']
model3 = sm.OLS(y, X).fit()
print(model3.summary())

# Regression 4: % chg homeless = social per capita
X = df_no_na['soc_iss_per_cap']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = df_no_na['pct_chg_homeless']
model4 = sm.OLS(y, X).fit()
print(model4.summary())

df['sus_iss_per_cap_lag'] = df.groupby('state')['sus_iss_per_cap'].shift(1)

reg_5_and_9 = df.dropna()

# Regression 5: homeless per capita = sustainable per capita (lag 1 year)
X = reg_5_and_9['sus_iss_per_cap_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_5_and_9['homeless_per_cap']
model5 = sm.OLS(y, X).fit()
print(model5.summary())

df['homeless_per_cap_lag'] = df.groupby('state')['homeless_per_cap'].shift(1)

reg_6_and_8 = df.dropna()

# Regression 6: sustainable per capita = homeless per capita  (lag 1 yr)
X = reg_6_and_8['homeless_per_cap_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_6_and_8['sus_iss_per_cap']
model6 = sm.OLS(y, X).fit()
print(model6.summary())

df['soc_iss_per_cap_lag'] = df.groupby('state')['soc_iss_per_cap'].shift(1)

reg_7_and_11 = df.dropna()

# Regression 7: homeless per capita = social per capita (lag 1 yr)
X = reg_7_and_11['soc_iss_per_cap_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_7_and_11['homeless_per_cap']
model7 = sm.OLS(y, X).fit()
print(model7.summary())

# Regression 8: social per capita = homeless per capita (lag 1 yr)
X = reg_6_and_8['homeless_per_cap_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_6_and_8['soc_iss_per_cap']
model8 = sm.OLS(y, X).fit()
print(model8.summary())

# Regression 9: % chg homeless = sustainable per capita (lag 1 yr)
X = reg_5_and_9['sus_iss_per_cap_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_5_and_9['pct_chg_homeless']
model9 = sm.OLS(y, X).fit()
print(model9.summary())

df_no_na['pct_chg_homeless_lag'] = df_no_na.groupby('state')['pct_chg_homeless'].shift(1)

reg_10_and_12 = df_no_na.dropna()

# Regression 10: sustainable per capita = % chg homeless (lag 1 yr)
X = reg_10_and_12['pct_chg_homeless_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_10_and_12['sus_iss_per_cap']
model10 = sm.OLS(y, X).fit()
print(model10.summary())

# Regression 11: % chg homeless = social per capita (lag 1 yr)
X = reg_7_and_11['soc_iss_per_cap_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_7_and_11['pct_chg_homeless']
model11 = sm.OLS(y, X).fit()
print(model11.summary())

# Regression 12: # social per capita = % chg homeless (lag 1 yr)
X = reg_10_and_12['pct_chg_homeless_lag']
X = sm.add_constant(X)  # Add a constant column for the intercept
y = reg_10_and_12['soc_iss_per_cap']
model12 = sm.OLS(y, X).fit()
print(model12.summary())