import xarray as xr
import pandas as pd
import glob
import os
import zipfile
import tempfile # For handling temporary extraction

# --- Configuration ---
input_dir = "./data/temp"
output_dir = "./data/processed_era5_cmip/"
os.makedirs(output_dir, exist_ok=True)

start_year = 2022
start_month = 1
start_day = 1
end_year = 2022
end_month = 1
end_day = 1 # Inclusive
# --- End Configuration ---

# Updated cmip_era5_map to include 'source_type' ('accumulated' or 'instantaneous')
cmip_era5_map = {
    'cllvi': {'era_name': 'tclw', 'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'kg m-2', 'long_name': 'Total Column Condensate (Liquid Water Path)'},
    'clivi': {'era_name': 'tciw', 'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'kg m-2', 'long_name': 'Total Column Condensate (Ice Water Path)'},
    'tas':   {'era_name': 't2m',   'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'K', 'long_name': 'Near-Surface Air Temperature'},
    'uas':   {'era_name': 'u10',   'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'm s-1', 'long_name': 'Eastward Near-Surface Wind'},
    'vas':   {'era_name': 'v10',   'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'm s-1', 'long_name': 'Northward Near-Surface Wind'},
    'pres_msl': {'era_name': 'msl', 'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'Pa', 'long_name': 'Mean Sea Level Pressure'},
    'sst':   {'era_name': 'sst',   'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': 'K', 'long_name': 'Sea Surface Temperature'},
    'sic':   {'era_name': 'siconc',    'source_type': 'instantaneous', 'calc': lambda da: da, 'unit': '1', 'long_name': 'Sea Ice Area Fraction (0-1)'},
    'rlut': {
        'era_name': 'ttr', 'source_type': 'accumulated',
        'calc': lambda da: -da / 3600.0,
        'unit': 'W m-2', 'long_name': 'TOA Outgoing Longwave Radiation'
    },
    'rsds': {
        'era_name': 'ssrd', 'source_type': 'accumulated',
        'calc': lambda da: da / 3600.0,
        'unit': 'W m-2', 'long_name': 'Surface Downwelling Shortwave Radiation'
    },
    'pr': {
        'era_name': 'tp', 'source_type': 'accumulated',
        'calc': lambda da: (da * 1000.0) / 3600.0,
        'unit': 'kg m-2 s-1', 'long_name': 'Precipitation Rate'
    },
    # rsut requires two 'accumulated' variables: 'tisr' and 'tsr'
}

date_range = pd.date_range(
    start=f"{start_year}-{start_month}-{start_day}",
    end=f"{end_year}-{end_month}-{end_day}",
    freq='D'
)

for date_pd in date_range:
    year = date_pd.year
    month = date_pd.month
    day = date_pd.day

    input_zip_filename_pattern = os.path.join(input_dir, f"era5_variables_{year}{month:02d}{day:02d}.zip")
    # If your zip files are actually named .nc, use this pattern:
    # input_zip_filename_pattern = os.path.join(input_dir, f"era5_variables_{year}{month:02d}{day:02d}.nc")
    input_zip_files = glob.glob(input_zip_filename_pattern)

    if not input_zip_files:
        print(f"No input ZIP file found for {year}-{month:02d}-{day:02d} matching pattern {input_zip_filename_pattern}")
        continue

    input_zip_filepath = input_zip_files[0]
    print(f"Processing ZIP archive: {input_zip_filepath}")

    ds_accum_era = None
    ds_instant_era = None
    extracted_accum_name = None
    extracted_instant_name = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(input_zip_filepath, 'r') as zip_ref:
                extracted_files = []
                for member_name in zip_ref.namelist():
                    if member_name.endswith('.nc'):
                        zip_ref.extract(member_name, tmpdir)
                        extracted_path = os.path.join(tmpdir, member_name)
                        extracted_files.append(extracted_path)
                        print(f"  Extracted '{member_name}' to '{extracted_path}'")

                if not extracted_files:
                    print(f"  No .nc files found inside {input_zip_filepath}")
                    continue

                # Identify accum and instant files
                for ext_file in extracted_files:
                    if 'accum.nc' in os.path.basename(ext_file).lower(): # Check for 'accum.nc'
                        ds_accum_era = xr.open_dataset(ext_file)
                        extracted_accum_name = os.path.basename(ext_file)
                        print(f"  Identified and opened accumulated data: {extracted_accum_name}")
                    elif 'instant.nc' in os.path.basename(ext_file).lower(): # Check for 'instant.nc'
                        ds_instant_era = xr.open_dataset(ext_file)
                        extracted_instant_name = os.path.basename(ext_file)
                        print(f"  Identified and opened instantaneous data: {extracted_instant_name}")
            
            if not ds_accum_era and not ds_instant_era:
                 print(f"  Could not identify specific accum/instant .nc files based on name in {input_zip_filepath}. Please check file names within zip.")
                 # Fallback: if only one nc file was extracted, try to use it for all.
                 if len(extracted_files) == 1:
                    print(f"  Found only one .nc file: {extracted_files[0]}. Attempting to use it for all variables (may lead to errors).")
                    # This is a simple fallback; user might need to refine logic if names are very different
                    ds_fallback_era = xr.open_dataset(extracted_files[0])
                    ds_accum_era = ds_fallback_era 
                    ds_instant_era = ds_fallback_era
                 else:
                    continue


            # --- Start Processing ---
            ds_cmip = xr.Dataset()

            for cmip_name, props in cmip_era5_map.items():
                current_ds_era = None
                source_file_type_name = ""

                if props['source_type'] == 'accumulated':
                    current_ds_era = ds_accum_era
                    source_file_type_name = extracted_accum_name or "Accumulated Data File"
                elif props['source_type'] == 'instantaneous':
                    current_ds_era = ds_instant_era
                    source_file_type_name = extracted_instant_name or "Instantaneous Data File"

                if current_ds_era is None:
                    print(f"    Warning: Source dataset for type '{props['source_type']}' (needed for {cmip_name}) not available.")
                    continue

                if props['era_name'] in current_ds_era:
                    print(f"    Processing {cmip_name} from {props['era_name']} (source: {props['source_type']})...")
                    data_array = props['calc'](current_ds_era[props['era_name']])
                    data_array.name = cmip_name
                    data_array.attrs['units'] = props['unit']
                    data_array.attrs['long_name'] = props['long_name']
                    data_array.attrs['standard_name'] = cmip_name
                    ds_cmip[cmip_name] = data_array
                else:
                    print(f"    Warning: ERA5 variable {props['era_name']} for {cmip_name} not found in {source_file_type_name}")

            # Special handling for 'rsut' (needs 'tisr' and 'tsr' from accumulated data)
            if ds_accum_era:
                if 'tisr' in ds_accum_era and 'tsr' in ds_accum_era:
                    print("    Processing rsut from tisr and tsr (source: accumulated)...")
                    rsut_da = (ds_accum_era['tisr'] - ds_accum_era['tsr']) / 3600.0
                    rsut_da.name = 'rsut'
                    rsut_da.attrs['units'] = 'W m-2'
                    rsut_da.attrs['long_name'] = 'TOA Outgoing Shortwave Radiation'
                    rsut_da.attrs['standard_name'] = 'rsut'
                    ds_cmip['rsut'] = rsut_da
                else:
                    print(f"    Warning: ERA5 variables 'tisr' and/or 'tsr' for rsut not found in {extracted_accum_name or 'Accumulated Data File'}")
            else:
                 print(f"    Warning: Accumulated data file not available for 'rsut' processing.")


            if ds_cmip.data_vars:
                output_filename = f"era5_cmip_processed_{year}{month:02d}{day:02d}.nc"
                output_filepath = os.path.join(output_dir, output_filename)

                ds_cmip.attrs['Conventions'] = 'CF-1.8'
                ds_cmip.attrs['history'] = f"Processed from ERA5 data (extracted from zip) on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ds_cmip.attrs['source_zip_file'] = os.path.basename(input_zip_filepath)
                if extracted_accum_name: ds_cmip.attrs['source_accum_nc_file'] = extracted_accum_name
                if extracted_instant_name: ds_cmip.attrs['source_instant_nc_file'] = extracted_instant_name


                print(f"  Saving processed file to: {output_filepath}")
                ds_cmip.to_netcdf(output_filepath)
            else:
                print(f"  No variables processed from {input_zip_filepath}, skipping save.")

            # Close datasets if they were opened
            if ds_accum_era: ds_accum_era.close()
            if ds_instant_era and ds_instant_era is not ds_accum_era: # Avoid double close if fallback was used
                 ds_instant_era.close()


    except zipfile.BadZipFile:
        print(f"Error: File {input_zip_filepath} is not a valid ZIP archive or is corrupted.")
    except Exception as e:
        print(f"An error occurred while processing {input_zip_filepath}: {e}")
    # Temporary directory and its contents are automatically cleaned up here

print("\nProcessing complete.")
print("--- Notes on Units and Calculations ---")
print("Temperatures (tas, sst): Kept in Kelvin (K).")
print("Pressures (pres_msl): Kept in Pascals (Pa).")
print("Radiation fluxes (rlut, rsut, rsds): Converted from accumulated J m-2 to W m-2 (average rate over the hour).")
print("  rlut = - (top_net_thermal_radiation / 3600)")
print("  rsut = (toa_incident_solar_radiation - top_net_solar_radiation) / 3600")
print("  rsds = surface_solar_radiation_downwards / 3600")
print("Precipitation (pr): Converted from accumulated meters to kg m-2 s-1.")
print("  pr = (total_precipitation_meters * 1000_kg_m-3) / 3600_s")
print("Sea Ice Cover (sic): Kept as a fraction (0-1). Multiply by 100 if percentage is needed.")