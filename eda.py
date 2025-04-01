import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure proper year formatting
merged_df['year'] = merged_df['year'].astype(int)

# ----------------------------
# 1. Create DALYs Metric
# ----------------------------
merged_df['DALYs'] = merged_df[['Deaths', 'YLLs (Years of Life Lost)', 'YLDs (Years Lived with Disability)']].sum(axis=1)

# ----------------------------
# 2. Missing Data Summary
# ----------------------------
missing_data_pct = (merged_df.isna().mean() * 100).sort_values(ascending=False)

# ----------------------------
# 3. Correlation Matrix
# ----------------------------
eda_numeric = merged_df.select_dtypes(include=['float64', 'int64'])
eda_corr = eda_numeric.corr()

# ----------------------------
# 4. COPD Burden by Sub-Region
# ----------------------------
regional_burden = merged_df.groupby('Sub-Region')[
    ['Deaths', 'YLLs (Years of Life Lost)', 'YLDs (Years Lived with Disability)']
].mean().sort_values(by='Deaths', ascending=False)

# ----------------------------
# 5. Top Countries by COPD Burden
# ----------------------------
top_countries_copd = merged_df.groupby('country')[
    ['Deaths', 'YLLs (Years of Life Lost)', 'Smoking', 'Ambient particulate matter pollution']
].mean().sort_values(by='Deaths', ascending=False).head(10)

# ----------------------------
# 6. Regional Time-Series Analysis
# ----------------------------
region_features = [
    'Sub-Region', 'year', 'GDP PER CAPITA (USD)', 'Population Density',
    'Total CO2 Emission excluding LUCF (Mt)', 'Ambient particulate matter pollution',
    'Smoking', 'Household air pollution from solid fuels',
    'Deaths', 'YLLs (Years of Life Lost)', 'YLDs (Years Lived with Disability)', 'DALYs'
]
regional_df = merged_df[region_features].copy()
regional_trends = regional_df.groupby(['Sub-Region', 'year']).mean(numeric_only=True).reset_index()

# ----------------------------
# 7. Time-Lag Analysis (5-Year Lag for PM)
# ----------------------------
merged_df_sorted = merged_df.sort_values(by=['Sub-Region', 'year'])
merged_df_sorted['PM_Lag5'] = merged_df_sorted.groupby('Sub-Region')[
    'Ambient particulate matter pollution'
].shift(5)
lag_corr = merged_df_sorted[['PM_Lag5', 'DALYs']].corr().iloc[0, 1]

# ----------------------------
# 8. Visualizations
# ----------------------------
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 25))
axes = axes.flatten()

plot_features = [
    'GDP PER CAPITA (USD)', 'Population Density',
    'Total CO2 Emission excluding LUCF (Mt)', 'Ambient particulate matter pollution',
    'Smoking', 'Household air pollution from solid fuels',
    'Deaths', 'YLLs (Years of Life Lost)', 'YLDs (Years Lived with Disability)', 'DALYs'
]
regions = merged_df['Sub-Region'].dropna().unique()

for i, feature in enumerate(plot_features):
    ax = axes[i]
    for region in regions:
        data = regional_trends[regional_trends['Sub-Region'] == region]
        sns.lineplot(data=data, x='year', y=feature, label=region, ax=ax)
    ax.set_title(f"{feature} Over Time by Region")
    ax.set_ylabel("Mean Value")
    ax.set_xlabel("Year")
    ax.legend(loc='best')

plt.tight_layout()
plt.show()

# ----------------------------
# 9. Lag Effect Scatterplot
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df_sorted, x='PM_Lag5', y='DALYs', hue='Sub-Region')
plt.title("DALYs vs. PM Pollution (Lagged 5 Years)")
plt.xlabel("Ambient Particulate Matter Pollution (5-Year Lag)")
plt.ylabel("DALYs")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 10. Additional Summary Tables
# ----------------------------
latest_year = regional_trends['year'].max()
latest_summary = regional_trends[regional_trends['year'] == latest_year].set_index('Sub-Region')
lag_corr_df = pd.DataFrame({'PM Lag 5-Year Correlation with DALYs': [lag_corr]})

import ace_tools as tools
tools.display_dataframe_to_user(name="Regional Comparison Summary (Latest Year)", dataframe=latest_summary)
tools.display_dataframe_to_user(name="Lag Correlation Summary", dataframe=lag_corr_df)
tools.display_dataframe_to_user(name="Missing Data Summary (%)", dataframe=missing_data_pct.to_frame(name="Missing %"))




import numpy as np

# Ensure year and population are numeric
merged_df['year'] = merged_df['year'].astype(int)
merged_df['Population'] = pd.to_numeric(merged_df['Population'], errors='coerce')

# =============================================
# 1. Histograms/KDE plots for variable shapes
# =============================================
features_to_plot = [
    'GDP PER CAPITA (USD)', 'Population Density',
    'Total CO2 Emission excluding LUCF (Mt)', 'Ambient particulate matter pollution',
    'Smoking', 'Household air pollution from solid fuels',
    'Deaths', 'YLLs (Years of Life Lost)', 'YLDs (Years Lived with Disability)', 'DALYs'
]

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18, 20))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    ax = axes[i]
    sns.histplot(merged_df[feature].dropna(), kde=True, ax=ax, bins=30)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel('')
plt.tight_layout()
plt.show()

# =============================================
# 2. Normalize COPD Burden per 100,000
# =============================================
merged_df['DALYs_per_100k'] = (merged_df['DALYs'] / merged_df['Population']) * 100000
merged_df['Deaths_per_100k'] = (merged_df['Deaths'] / merged_df['Population']) * 100000

# =============================================
# 3. Rate of Change (2000–2020) by Region
# =============================================
rate_features = ['DALYs_per_100k', 'Deaths_per_100k', 'Ambient particulate matter pollution', 'Smoking']
change_df = merged_df[merged_df['year'].isin([2000, 2020])].copy()
region_changes = change_df.groupby(['Sub-Region', 'year'])[rate_features].mean().unstack().T
region_change_pct = ((region_changes[2020] - region_changes[2000]) / region_changes[2000]) * 100

# =============================================
# 4. Stratified Comparison: High vs Low Pollution
# =============================================
pollution_median = merged_df['Ambient particulate matter pollution'].median()
merged_df['Pollution_Group'] = np.where(
    merged_df['Ambient particulate matter pollution'] > pollution_median, 'High PM', 'Low PM'
)
stratified_summary = merged_df.groupby('Pollution_Group')[
    ['DALYs_per_100k', 'Deaths_per_100k', 'Smoking', 'Household air pollution from solid fuels']
].mean()

# =============================================
# 5. Boxplots of DALYs by Region & Pollution Level
# =============================================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=merged_df, x='Sub-Region', y='DALYs_per_100k')
plt.xticks(rotation=45)
plt.title('DALYs per 100k by Sub-Region')

plt.subplot(1, 2, 2)
sns.boxplot(data=merged_df, x='Pollution_Group', y='DALYs_per_100k')
plt.title('DALYs per 100k by Pollution Level')
plt.tight_layout()
plt.show()

# =============================================
# Export Rate of Change & Stratified Summary
# =============================================
import ace_tools as tools
tools.display_dataframe_to_user(name="Rate of Change by Region (2000–2020, %)", dataframe=region_change_pct)
tools.display_dataframe_to_user(name="Stratified Summary: High vs Low Pollution", dataframe=stratified_summary)


# Dynamically detect earliest and latest year in dataset
min_year = merged_df['year'].min()
max_year = merged_df['year'].max()

# Updated rate of change computation
change_df = merged_df[merged_df['year'].isin([min_year, max_year])].copy()
region_changes = change_df.groupby(['Sub-Region', 'year'])[
    ['DALYs_per_100k', 'Deaths_per_100k', 'Ambient particulate matter pollution', 'Smoking']
].mean().unstack().T

region_change_pct = ((region_changes[max_year] - region_changes[min_year]) / region_changes[min_year]) * 100

# Export corrected summaries
import ace_tools as tools
tools.display_dataframe_to_user(name=f"Rate of Change by Region ({min_year}–{max_year}, %)", dataframe=region_change_pct)


# Identify valid regions with data for both min and max year
valid_regions = change_df.groupby(['Sub-Region', 'year']).size().unstack().dropna().index.tolist()

# Filter data to those regions
change_df_filtered = change_df[change_df['Sub-Region'].isin(valid_regions)]

# Compute mean values for each year
region_changes_filtered = change_df_filtered.groupby(['Sub-Region', 'year'])[
    ['DALYs_per_100k', 'Deaths_per_100k', 'Ambient particulate matter pollution', 'Smoking']
].mean().unstack().T

# Compute percent change between min and max year
region_change_pct_filtered = (
    (region_changes_filtered[max_year] - region_changes_filtered[min_year]) /
    region_changes_filtered[min_year]
) * 100

# Display cleaned percent change results
import ace_tools as tools
tools.display_dataframe_to_user(
    name=f"Cleaned Rate of Change by Region ({min_year}–{max_year}, %)",
    dataframe=region_change_pct_filtered
)


# Reset and clean multiindex columns from unstack().T
region_changes_flat = region_changes_filtered.copy()
region_changes_flat.columns = region_changes_flat.columns.map(lambda x: f"{x[0]}_{x[1]}")

# Build percent change DataFrame manually
change_pct_dict = {}
for region in valid_regions:
    try:
        before = region_changes_flat[f'{min_year}_{region}']
        after = region_changes_flat[f'{max_year}_{region}']
        pct_change = ((after - before) / before) * 100
        change_pct_dict[region] = pct_change
    except KeyError:
        continue

# Convert to DataFrame
region_change_pct_clean = pd.DataFrame(change_pct_dict).T
region_change_pct_clean.index.name = "Sub-Region"

# Display clean output
import ace_tools as tools
tools.display_dataframe_to_user(
    name=f"Regional % Change in Indicators ({min_year}–{max_year})",
    dataframe=region_change_pct_clean
)


# Identify years available per region
year_region_counts = merged_df.groupby(['Sub-Region', 'year']).size().unstack()

# Find years with complete data across all regions
valid_years = year_region_counts.dropna(axis=1, how='any').columns.tolist()

# Proceed only if at least two valid years exist
if len(valid_years) >= 2:
    first_valid_year = valid_years[0]
    last_valid_year = valid_years[-1]

    # Filter data for only these two years
    change_df_final = merged_df[merged_df['year'].isin([first_valid_year, last_valid_year])].copy()

    # Compute mean indicators per region and year
    region_changes_final = change_df_final.groupby(['Sub-Region', 'year'])[
        ['DALYs_per_100k', 'Deaths_per_100k', 'Ambient particulate matter pollution', 'Smoking']
    ].mean().unstack().T

    # Reformat columns for safe access
    region_changes_final.columns = region_changes_final.columns.map(lambda x: f"{x[0]}_{x[1]}")

    # Calculate % change manually
    change_final_dict = {}
    for region in merged_df['Sub-Region'].dropna().unique():
        try:
            before = region_changes_final[f'{first_valid_year}_{region}']
            after = region_changes_final[f'{last_valid_year}_{region}']
            pct_change = ((after - before) / before) * 100
            change_final_dict[region] = pct_change
        except KeyError:
            continue

    # Convert to DataFrame
    region_change_final_df = pd.DataFrame(change_final_dict).T
    region_change_final_df.index.name = f"Sub-Region ({first_valid_year}–{last_valid_year})"

    # Display final result
    import ace_tools as tools
    tools.display_dataframe_to_user(
        name=f"Regional % Change in Indicators ({first_valid_year}–{last_valid_year})",
        dataframe=region_change_final_df
    )
else:
    region_change_final_df = pd.DataFrame()
    region_change_final_df["error"] = ["Insufficient overlapping years for all sub-regions"]
    import ace_tools as tools
    tools.display_dataframe_to_user(
        name="Regional Change Error",
        dataframe=region_change_final_df
    )



correlation heatmap
# Compute correlation matrix again just to ensure freshness
eda_numeric = merged_df.select_dtypes(include=['float64', 'int64'])
eda_corr = eda_numeric.corr()

# Optional: focus only on features most correlated with DALYs
top_corr_features = eda_corr['DALYs'].abs().sort_values(ascending=False).head(15).index.tolist()
top_corr_matrix = eda_corr.loc[top_corr_features, top_corr_features]

# Plot heatmap with annotations
plt.figure(figsize=(12, 10))
sns.heatmap(top_corr_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5)
plt.title("Top 15 Features Most Correlated with DALYs")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


