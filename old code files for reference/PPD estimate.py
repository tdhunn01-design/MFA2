import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import os

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print('Loading data...')
df = pd.read_excel('PPD estimate.xlsx')
df.columns = ['Year', 'PPD']
df_clean = df.dropna(subset=['PPD'])
print(f"Loaded {len(df_clean)} data points")

# =============================================================================
# 2. MODEL PARAMETERS
# =============================================================================
interp_start = 1850
interp_end = 2039

# Split data into within-range and outside-range
data_within_range = df_clean[(df_clean['Year'] >= interp_start) & (df_clean['Year'] <= interp_end)]
data_before = df_clean[df_clean['Year'] < interp_start]
data_after = df_clean[df_clean['Year'] > interp_end]

print(f"\n📊 Data summary:")
print(f"  Data points within {interp_start}-{interp_end}: {len(data_within_range)}")
print(f"  Data points before {interp_start}: {len(data_before)}")
print(f"  Data points after {interp_end}: {len(data_after)}")

# =============================================================================
# 3. CREATE INTERPOLATION MODEL (CUBIC SPLINE)
# =============================================================================
if len(data_within_range) >= 3:
    years_with_data = data_within_range['Year'].values
    ppd_with_data = data_within_range['PPD'].values
    cs = CubicSpline(years_with_data, ppd_with_data, bc_type='natural')
    print(f"✅ Cubic spline created using {len(data_within_range)} data points")
else:
    print("⚠️  Not enough data points for cubic spline")
    cs = None

# =============================================================================
# 4. CREATE EXTRAPOLATION MODELS (LINEAR TRENDS)
# =============================================================================
# Backward trend: from 5.0 in 1850 to 6.0 in 1600
slope_backward = (5.0 - 6.0) / (1850 - 1600)  # = -1.0 / 250 = -0.004
intercept_backward = 5.0 - slope_backward * 1850

# Forward trend: from 2.32 in 2039 to 1.9 in 2100
slope_forward = (1.9 - 2.32) / (2100 - 2039)  # = -0.42 / 61 = -0.006885
intercept_forward = 2.32 - slope_forward * 2039

print(f"\n📐 Linear trend equations:")
print(f"  Backward (1600-1850): PPD = {slope_backward:.6f} × Year + {intercept_backward:.2f}")
print(f"  Forward (2039-2100): PPD = {slope_forward:.6f} × Year + {intercept_forward:.2f}")

# =============================================================================
# 5. GENERATE PROJECTIONS FOR ALL YEARS
# =============================================================================
print(f"\n⏳ Generating projections for 1600-2100...")
all_years = np.arange(1600, 2101, 1)
ppd_combined = np.zeros_like(all_years, dtype=float)

for i, year in enumerate(all_years):
    if interp_start <= year <= interp_end and cs is not None:
        ppd_combined[i] = cs(year)  # Interpolation
    elif year < interp_start:
        ppd_combined[i] = slope_backward * year + intercept_backward  # Backward extrapolation
    else:  # year > interp_end
        ppd_combined[i] = slope_forward * year + intercept_forward  # Forward extrapolation

print(f"✅ Generated {len(all_years)} projections (years {all_years.min()}-{all_years.max()})")

# =============================================================================
# 6. SAVE PROJECTIONS TO CSV (DO THIS EARLY TO ENSURE DATA IS SAVED)
# =============================================================================
print(f"\n💾 Saving projections to CSV...")
projected_df = pd.DataFrame({
    'Year': all_years,
    'PPD_Projected': ppd_combined,
    'Method': ['Linear (backward)' if y < interp_start else 
               'Linear (forward)' if y > interp_end else 
               'Cubic Spline' for y in all_years]
})

output_file = 'spain_ppd_projections_1600_2100.csv'
projected_df.to_csv(output_file, index=False)
print(f"✅ Projections saved to '{output_file}'")
print(f"📊 File contains {len(projected_df)} rows")

# Show sample of saved data
print("\n📋 Sample of saved data (first 5 rows):")
print(projected_df.head())
print("\n📋 Sample of saved data (last 5 rows):")
print(projected_df.tail())

# =============================================================================
# 7. CREATE VISUALIZATION
# =============================================================================
print(f"\n🎨 Creating plot...")
plt.figure(figsize=(15, 8))

# Original data points
plt.scatter(df_clean['Year'], df_clean['PPD'], 
           color='blue', s=60, alpha=0.8, label='Available PPD data', 
           zorder=4, edgecolors='white', linewidth=0.5)

# Combined curve
plt.plot(all_years, ppd_combined, 'r-', linewidth=2.5, 
         label='Combined model', alpha=0.9, zorder=2)

# Add vertical lines to mark the interpolation range
plt.axvline(x=interp_start, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
plt.axvline(x=interp_end, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

# Add horizontal lines for target values
plt.axhline(y=6.0, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='Target: 6.0 in 1600')
plt.axhline(y=5.0, color='orange', linestyle=':', alpha=0.5, linewidth=1)
plt.axhline(y=2.32, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Target: 2.32 in 2039')
plt.axhline(y=1.9, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Target: 1.9 in 2100')

# Add text labels for the regions
plt.text(interp_start + 90, plt.ylim()[1] * 0.9, 
         'Cubic Spline\n(Interpolation 1850-2039)', 
         ha='center', fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.4))
plt.text(1700, plt.ylim()[1] * 0.7, 
         'Linear Trend\n(1600-1850)\n6.0 → 5.0', 
         ha='center', fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.4))
plt.text(2070, plt.ylim()[1] * 0.7, 
         'Linear Trend\n(2039-2100)\n2.32 → 1.9', 
         ha='center', fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.4))

# Customize the plot
plt.title('Spain: Persons Per Dwelling (1600-2100)\nCubic Spline (1850-2039) + Custom Linear Trends', 
          fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Persons Per Dwelling', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=10)
plt.xlim(1600, 2100)

# Adjust y-axis limits
y_min = min(ppd_combined.min(), df_clean['PPD'].min(), 1.8) - 0.2
y_max = max(ppd_combined.max(), df_clean['PPD'].max(), 6.1) + 0.2
plt.ylim(y_min, y_max)

# Set x-ticks for key years
plt.xticks([1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2039, 2050, 2100], rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# 8. PRINT SUMMARY STATISTICS
# =============================================================================
print("\n📈 PROJECTIONS FOR KEY YEARS")
print("=" * 50)
key_years = [1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2039, 2050, 2075, 2100]
for year in key_years:
    if year in all_years:
        idx = np.where(all_years == year)[0][0]
        if year < interp_start:
            method = "Linear (backward)"
        elif year > interp_end:
            method = "Linear (forward)"
        else:
            method = "Cubic Spline"
        print(f"  Year {year}: {ppd_combined[idx]:.2f} PPD [{method}]")

# Target verification
print(f"\n✅ Target verification:")
print(f"  1600: {ppd_combined[1600-1600]:.2f} PPD (target: 6.0)")
print(f"  1850: {ppd_combined[1850-1600]:.2f} PPD (target: 5.0)")
print(f"  2039: {ppd_combined[2039-1600]:.2f} PPD (target: 2.32)")
print(f"  2100: {ppd_combined[2100-1600]:.2f} PPD (target: 1.9)")

print(f"\n✨ Done! CSV file saved and plot displayed.")