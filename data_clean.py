
import pandas as pd

# File paths (update as needed)
file1_path = "IHME-GBD_2021_DATA-5503ed84-1.csv"  # Risk factors
file2_path = "IHME-GBD_2021_DATA-6f8f7ceb-1.csv"  # Cause of death

file_air_pollution = "air-pollution.csv"
file_co2 = "co2 Emission Africa.csv"
file_haq = "IHME_GBD_2019_HAQ_1990_2019_DATA.CSV"

# Load datasets
df_risk = pd.read_csv(file1_path)
df_cause = pd.read_csv(file2_path)
df_air = pd.read_csv(file_air_pollution)
df_co2 = pd.read_csv(file_co2)
df_haq = pd.read_csv(file_haq)

# Dataset 1: Health Burden (Cause of Death Only)
df_health_burden = df_cause.drop(columns=['upper', 'lower'])
df_health_burden['type'] = 'cause'
df_health_burden['risk_factor'] = None

# Dataset 2: Risk Factor Burden
df_risk_factor = df_risk.drop(columns=['upper', 'lower'])
df_risk_factor = df_risk_factor.rename(columns={'rei': 'risk_factor'})
df_risk_factor['type'] = 'risk'

# Dataset 3: Contextual Dataset
african_countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
    'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros',
    'Congo', 'Côte d’Ivoire', 'Democratic Republic of the Congo', 'Djibouti',
    'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon',
    'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia',
    'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco',
    'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe',
    'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
    'South Sudan', 'Sudan', 'Togo', 'Tunisia', 'Uganda', 'United Republic of Tanzania',
    'Zambia', 'Zimbabwe'
]

df_haq_filtered = df_haq[
    (df_haq['indicator_name'] == 'Chronic respiratory diseases') &
    (df_haq['location_name'].isin(african_countries)) &
    (df_haq['year_id'] >= 2000)
][['location_name', 'year_id', 'val']].rename(columns={
    'location_name': 'Country',
    'year_id': 'Year',
    'val': 'HAQ_Chronic_Respiratory'
})

df_air_africa = df_air[(df_air['Country'].isin(african_countries)) & (df_air['Year'] >= 2000)]
df_co2_africa = df_co2[(df_co2['Country'].isin(african_countries)) & (df_co2['Year'] >= 2000)]

df_contextual = pd.merge(df_air_africa, df_co2_africa, on=['Country', 'Year'], how='outer')
df_contextual = pd.merge(df_contextual, df_haq_filtered, on=['Country', 'Year'], how='left')
df_contextual = df_contextual.loc[:, ~df_contextual.columns.str.contains('upper|lower')]

# Save all datasets to CSV
df_health_burden.to_csv("health_burden_dataset.csv", index=False)
df_risk_factor.to_csv("risk_factor_dataset.csv", index=False)
df_contextual.to_csv("contextual_dataset.csv", index=False)
"""

# Write to file
file_path = "/mnt/data/build_clean_datasets.py"
with open(file_path, "w") as f:
    f.write(python_code)

file_path


# Create a dictionary mapping column names to their respective units
unit_mapping = {
    'Country': 'Country Name',
    'Year': 'Year',
    'Nitrogen Oxide': 'Kilotonnes/year',
    'Sulphur Dioxide': 'Kilotonnes/year',
    'Carbon Monoxide': 'Kilotonnes/year',
    'Organic Carbon': 'Kilotonnes/year',
    'NMVOCs': 'Kilotonnes/year',
    'Black Carbon': 'Kilotonnes/year',
    'Ammonia': 'Kilotonnes/year',
    'Sub-Region': 'Text',
    'Code': 'ISO Code',
    'Population': 'People',
    'GDP PER CAPITA (USD)': 'USD',
    'GDP PER CAPITA PPP (USD)': 'USD (PPP adjusted)',
    'Area (Km2)': 'Square Kilometers',
    'Transportation (Mt)': 'Megatonnes CO₂/year',
    'Total CO2 Emission including LUCF (Mt)': 'Megatonnes CO₂/year',
    'Total CO2 Emission excluding LUCF (Mt)': 'Megatonnes CO₂/year',
    'Other Fuel Combustion (Mt)': 'Megatonnes CO₂/year',
    'Manufacturing/Construction (Mt)': 'Megatonnes CO₂/year',
    'Land-Use Change and Forestry (Mt)': 'Megatonnes CO₂/year',
    'Industrial Processes (Mt)': 'Megatonnes CO₂/year',
    'Fugitive Emissions (Mt)': 'Megatonnes CO₂/year',
    'Energy (Mt)': 'Megatonnes CO₂/year',
    'Electricity/Heat (Mt)': 'Megatonnes CO₂/year',
    'Bunker Fuels (Mt)': 'Megatonnes CO₂/year',
    'Building (Mt)': 'Megatonnes CO₂/year',
    'HAQ_Chronic_Respiratory': 'Index (0–100)'
}

# Convert unit mapping to a DataFrame
df_unit_mapping = pd.DataFrame(list(unit_mapping.items()), columns=["Column", "Unit"])

# Save as a CSV file
unit_file_path = "/mnt/data/contextual_dataset_units.csv"
df_unit_mapping.to_csv(unit_file_path, index=False)

unit_file_path
