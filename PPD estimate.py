import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel file
# Replace 'your_file.xlsx' with the actual path to your Excel file
df = pd.read_excel('PPD estimate.xlsx')

# Assuming first column is years and second column is PPD
# Let's name the columns for clarity
df.columns = ['Year', 'PPD']

# Remove rows where PPD is missing (NaN)
df_clean = df.dropna(subset=['PPD'])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot only the available data points as dots
plt.scatter(df_clean['Year'], df_clean['PPD'], 
           color='blue', s=50, alpha=0.7, label='Available PPD data')

# Customize the plot
plt.title('Spain: Persons Per Dwelling (1850-2050)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Persons Per Dwelling', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Set x-axis limits to show the full range
plt.xlim(1850, 2050)

# Add some padding to y-axis
y_min = df_clean['PPD'].min() - 0.5
y_max = df_clean['PPD'].max() + 0.5
plt.ylim(y_min, y_max)

# Add minor gridlines
plt.grid(True, which='both', alpha=0.2)
plt.minorticks_on()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Optional: Print the available data points
print("\nAvailable PPD Data Points:")
print("=" * 40)
for index, row in df_clean.iterrows():
    print(f"Year: {int(row['Year']):4d} | PPD: {row['PPD']:.2f}")

# Optional: Print summary statistics
print("\n" + "=" * 40)
print(f"Total data points: {len(df_clean)}")
print(f"Year range: {int(df_clean['Year'].min())} to {int(df_clean['Year'].max())}")
print(f"PPD range: {df_clean['PPD'].min():.2f} to {df_clean['PPD'].max():.2f}")
print(f"Average PPD: {df_clean['PPD'].mean():.2f}")


from scipy.interpolate import CubicSpline
import numpy as np

# Create cubic spline interpolation
# Use only the years where we have data
years_with_data = df_clean['Year'].values
ppd_with_data = df_clean['PPD'].values

# Create cubic spline
cs = CubicSpline(years_with_data, ppd_with_data, bc_type='natural')

# Generate values for every year from 1850 to 2050
all_years = np.arange(1850, 2051, 1)
ppd_smooth = cs(all_years)

# Add to existing plot
plt.figure(figsize=(12, 6))

# Original data points
plt.scatter(df_clean['Year'], df_clean['PPD'], 
           color='blue', s=50, alpha=0.7, label='Available PPD data', zorder=3)

# Cubic spline curve
plt.plot(all_years, ppd_smooth, 'r-', linewidth=2, 
         label='Cubic spline interpolation', alpha=0.8, zorder=2)

# Customize the plot
plt.title('Spain: Persons Per Dwelling (1850-2050) with Cubic Spline', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Persons Per Dwelling', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(1850, 2050)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print projected values for key years
print("\n📈 CUBIC SPLINE PROJECTIONS FOR KEY YEARS")
print("=" * 50)
key_years = [1850, 1900, 1950, 2000, 2025, 2050]
for year in key_years:
    if year in all_years:
        print(f"Year {year}: {ppd_smooth[year-1850]:.2f} PPD")

# Optional: Create a dataframe with all projected values
projected_df = pd.DataFrame({
    'Year': all_years,
    'PPD_Projected': ppd_smooth
})

# Create a dataframe with all projected values
projected_df = pd.DataFrame({
    'Year': all_years,
    'PPD_Projected': ppd_smooth
})

# Save to CSV
projected_df.to_csv('spain_ppd_projections_1850_2050.csv', index=False)
print("\n✅ Projected data saved to 'spain_ppd_projections_1850_2050.csv'")

# Show first few rows
print("\n📊 First 10 rows of projected data:")
print(projected_df.head(10))

# Show last few rows
print("\n📊 Last 10 rows of projected data:")
print(projected_df.tail(10))