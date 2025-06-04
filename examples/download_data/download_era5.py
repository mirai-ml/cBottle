#%%
import cdsapi
import calendar # For iterating through days in a month

c = cdsapi.Client()

# --- User Defined Parameters ---
start_year = 2022
start_month = 1
start_day = 1

end_year = 2022
end_month = 1
end_day = 1 # Inclusive

# Geographical area [North, West, South, East]
# Example: Europe
# area = [70, -10, 30, 40]
# Example: Global (can be very large, be careful!)
area = [90, -180, -90, 180]
# Example: Your location (Palaiseau, approx) - for a small test
# area = [48.8, 2.2, 48.7, 2.3]


output_filename_template = "era5_variables_{year}{month:02d}{day:02d}.grib"
# --- End User Defined Parameters ---

# Corrected list of ERA5 variable names
# Ensure these are the correct API names from the CDS catalogue for 'reanalysis-era5-single-levels'
era5_variables = [
    'total_column_cloud_liquid_water',
    'total_column_cloud_ice_water',
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'top_net_thermal_radiation',        # This is accumulated. rlut = -top_net_thermal_radiation.
    'top_net_solar_radiation',          # This is accumulated. Used to derive rsut.
    'toa_incident_solar_radiation',     # This is accumulated. Used to derive rsut.
    'mean_sea_level_pressure',
    'total_precipitation',              # This is accumulated.
    'surface_solar_radiation_downwards',# This is accumulated.
    'sea_surface_temperature',
    'sea_ice_cover',
]

# Generate a list of all hours
times = [f"{h:02d}:00" for h in range(24)]

# Loop through years, months, and days
current_year = start_year
current_month = start_month
current_day = start_day

while True:
    date_str = f"{current_year}-{current_month:02d}-{current_day:02d}"
    output_file = output_filename_template.format(year=current_year, month=current_month, day=current_day)
    print(f"Requesting data for: {date_str}")
    print(f"Output file: {output_file}")

    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': era5_variables,
                'year': str(current_year),
                'month': f"{current_month:02d}",
                'day': f"{current_day:02d}",
                'time': times,
                'area': area, # North, West, South, East
                'format': 'grib',
            },
            output_file)
        print(f"Successfully downloaded {output_file}")
    except Exception as e:
        print(f"Error downloading data for {date_str}: {e}")

    # Increment day
    if current_year == end_year and current_month == end_month and current_day == end_day:
        break

    current_day += 1
    days_in_month = calendar.monthrange(current_year, current_month)[1]
    if current_day > days_in_month:
        current_day = 1
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

print("All requests completed.")

print("\n--- Important Notes on TOA Radiation Fluxes ---")
print("1. 'top_net_thermal_radiation' (tntr/ttr):")
print("   - This is an accumulated flux in J m⁻².")
print("   - It's defined as Incoming TOA LW - Outgoing TOA LW.")
print("   - Since Incoming TOA LW is ~0, rlut (Outgoing TOA LW) ≈ -top_net_thermal_radiation.")
print("   - To get W m⁻², divide the hourly accumulated J m⁻² by 3600.")
print("\n2. For 'rsut' (TOA Outgoing Shortwave Radiation):")
print("   - You downloaded 'toa_incident_solar_radiation' (tisr) and 'top_net_solar_radiation' (tnsr/tsr).")
print("   - Both are accumulated fluxes in J m⁻².")
print("   - rsut = toa_incident_solar_radiation - top_net_solar_radiation.")
print("   - To get W m⁻², perform the subtraction with the J m⁻² values, then divide the result by 3600.")
print("\n3. For other accumulated fluxes ('total_precipitation', 'surface_solar_radiation_downwards'):")
print("   - These are also accumulated (total_precipitation in m, surface_solar_radiation_downwards in J m⁻²).")
print("   - total_precipitation: The hourly value is the accumulation over the past hour. Multiply by 1000 for mm/hour.")
print("   - surface_solar_radiation_downwards: Divide J m⁻² by 3600 for average W m⁻² over the hour.")
# %%
