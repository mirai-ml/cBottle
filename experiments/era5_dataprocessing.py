#!/usr/bin/env python3
"""
Unified script to download ERA5 data, process to CMIP format, and convert to HEALPix.
"""
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import torch
import earth2grid
import os
import tempfile
import zipfile
from typing import Dict, List,  Optional
import sys
import argparse

# Configuration dictionaries for better extensibility
ERA5_VARIABLES = [
    'total_column_cloud_ice_water',
    'total_column_cloud_liquid_water',
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    'sea_surface_temperature',
    'total_precipitation',
    'surface_solar_radiation_downwards',
    'top_net_thermal_radiation',
    'top_net_solar_radiation',
    'toa_incident_solar_radiation',
    'sea_ice_cover',
]

CMIP_ERA5_MAP = {
    'cllvi': {
        'era_name': 'tclw',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'kg m-2',
        'long_name': 'Total Column Condensate (Liquid Water Path)'
    },
    'clivi': {
        'era_name': 'tciw',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'kg m-2',
        'long_name': 'Total Column Condensate (Ice Water Path)'
    },
    'tas': {
        'era_name': 't2m',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'K',
        'long_name': 'Near-Surface Air Temperature'
    },
    'uas': {
        'era_name': 'u10',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'm s-1',
        'long_name': 'Eastward Near-Surface Wind'
    },
    'vas': {
        'era_name': 'v10',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'm s-1',
        'long_name': 'Northward Near-Surface Wind'
    },
    'pres_msl': {
        'era_name': 'msl',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'Pa',
        'long_name': 'Mean Sea Level Pressure'
    },
    'sst': {
        'era_name': 'sst',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': 'K',
        'long_name': 'Sea Surface Temperature'
    },
    'sic': {
        'era_name': 'siconc',
        'source_type': 'instantaneous',
        'calc': lambda da: da,
        'unit': '1',
        'long_name': 'Sea Ice Area Fraction (0-1)'
    },
    'rlut': {
        'era_name': 'ttr',
        'source_type': 'accumulated',
        'calc': lambda da: -da / 3600.0,
        'unit': 'W m-2',
        'long_name': 'TOA Outgoing Longwave Radiation'
    },
    'rsds': {
        'era_name': 'ssrd',
        'source_type': 'accumulated',
        'calc': lambda da: da / 3600.0,
        'unit': 'W m-2',
        'long_name': 'Surface Downwelling Shortwave Radiation'
    },
    'pr': {
        'era_name': 'tp',
        'source_type': 'accumulated',
        'calc': lambda da: (da * 1000.0) / 3600.0,
        'unit': 'kg m-2 s-1',
        'long_name': 'Precipitation Rate'
    },
}

VARS_TO_KEEP = ['cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 'rsut', 'pres_msl', 'pr', 'rsds', 'sst', 'sic']


def download_era5_data(
    year: int,
    month: int,
    day: int,
    hour: str,
    output_path: str,
    variables: Optional[List[str]] = None
) -> str:
    """
    Download ERA5 data for a specific date and hour.

    Parameters:
    -----------
    year, month, day : int
        Date to download
    hour : str
        Hour in format "HH:MM" (e.g., "15:00")
    output_path : str
        Path where to save the downloaded file
    variables : Optional[List[str]]
        List of ERA5 variables to download. If None, uses default ERA5_VARIABLES

    Returns:
    --------
    str : Path to the downloaded NetCDF file

    Raises:
    -------
    Exception : If download fails
    """
    print(f"=== Step 1: Downloading ERA5 data ===")
    print(f"Date: {year}-{month:02d}-{day:02d}")
    print(f"Hour: {hour}")

    if variables is None:
        variables = ERA5_VARIABLES

    print(f"Variables: {len(variables)} variables")
    # Initialize CDS API client
    c = cdsapi.Client()

    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': str(year),
                'month': f"{month:02d}",
                'day': f"{day:02d}",
                'time': [hour],
                'data_format': 'netcdf',
                "download_format": "zip",
            },
            output_path
        )
        print(f"✓ Successfully downloaded ERA5 data to: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Error downloading ERA5 data: {e}")
        raise


def process_era5_to_cmip(
    input_download_file: str,
    output_path: str,
    cmip_era5_map: Optional[Dict] = None
) -> None:
    """
    Process ERA5 data from zip files to CMIP format.

    Parameters:
    -----------
    input_dir : str
        Directory containing ERA5 zip files
    output_dir : str
        Directory where processed CMIP files will be saved
    start_year, start_month, start_day : int
        Start date for processing
    end_year, end_month, end_day : int
        End date for processing (inclusive)
    cmip_era5_map : Optional[Dict]
        Mapping dictionary for CMIP variables. If None, uses default CMIP_ERA5_MAP
    """
    print(f"=== Step 2: Processing ERA5 data to CMIP format ===")
    print(f"Input file: {input_download_file}")
    print(f"Output file: {output_path}")
    # Create output directory if it doesn't exist

    if cmip_era5_map is None:
        cmip_era5_map = CMIP_ERA5_MAP

    input_zip_filepath = input_download_file
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
                        print(
                            f"  Extracted '{member_name}' to '{extracted_path}'")

                if not extracted_files:
                    raise Exception(
                        f"  No .nc files found inside {input_zip_filepath}")

                # Identify accum and instant files
                for ext_file in extracted_files:
                    if 'accum.nc' in os.path.basename(ext_file).lower():
                        ds_accum_era = xr.open_dataset(ext_file)
                        extracted_accum_name = os.path.basename(ext_file)
                        print(
                            f"  Identified and opened accumulated data: {extracted_accum_name}")
                    elif 'instant.nc' in os.path.basename(ext_file).lower():
                        ds_instant_era = xr.open_dataset(ext_file)
                        extracted_instant_name = os.path.basename(ext_file)
                        print(
                            f"  Identified and opened instantaneous data: {extracted_instant_name}")

            if not ds_accum_era and not ds_instant_era:
                print(
                    f"  Could not identify specific accum/instant .nc files based on name in {input_zip_filepath}")
                # Fallback: if only one nc file was extracted, try to use it for all
                if len(extracted_files) == 1:
                    print(
                        f"  Found only one .nc file: {extracted_files[0]}. Attempting to use it for all variables.")
                    ds_fallback_era = xr.open_dataset(extracted_files[0])
                    ds_accum_era = ds_fallback_era
                    ds_instant_era = ds_fallback_era
                else:
                    raise Exception(
                        f"  found 0 .nc files: {extracted_files}. Please check the file names.")

            # Start Processing
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
                    raise Exception(
                        f"  Source dataset for type '{props['source_type']}' (needed for {cmip_name}) not available.")

                if props['era_name'] in current_ds_era:
                    print(
                        f"    Processing {cmip_name} from {props['era_name']} (source: {props['source_type']})...")
                    data_array = props['calc'](
                        current_ds_era[props['era_name']])
                    data_array.name = cmip_name
                    data_array.attrs['units'] = props['unit']
                    data_array.attrs['long_name'] = props['long_name']
                    data_array.attrs['standard_name'] = cmip_name
                    ds_cmip[cmip_name] = data_array
                else:
                    raise Exception(
                        f"    Warning: ERA5 variable {props['era_name']} for {cmip_name} not found in {source_file_type_name}")

            # Special handling for 'rsut' (needs 'tisr' and 'tsr' from accumulated data)
            if ds_accum_era:
                if 'tisr' in ds_accum_era and 'tsr' in ds_accum_era:
                    print(
                        "    Processing rsut from tisr and tsr (source: accumulated)...")
                    rsut_da = (ds_accum_era['tisr'] -
                               ds_accum_era['tsr']) / 3600.0
                    rsut_da.name = 'rsut'
                    rsut_da.attrs['units'] = 'W m-2'
                    rsut_da.attrs['long_name'] = 'TOA Outgoing Shortwave Radiation'
                    rsut_da.attrs['standard_name'] = 'rsut'
                    ds_cmip['rsut'] = rsut_da
                else:
                    raise Exception(
                        f"    Warning: ERA5 variables 'tisr' and/or 'tsr' for rsut not found in {extracted_accum_name or 'Accumulated Data File'}")
            else:
                raise Exception(
                    f"    Warning: Accumulated data file not available for 'rsut' processing.")

            if ds_cmip.data_vars:
                ds_cmip.attrs['Conventions'] = 'CF-1.8'
                ds_cmip.attrs[
                    'history'] = f"Processed from ERA5 data (extracted from zip) on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ds_cmip.attrs['source_zip_file'] = os.path.basename(
                    input_zip_filepath)
                if extracted_accum_name:
                    ds_cmip.attrs['source_accum_nc_file'] = extracted_accum_name
                if extracted_instant_name:
                    ds_cmip.attrs['source_instant_nc_file'] = extracted_instant_name

                print(f"  Saving processed file to: {output_path}")
                ds_cmip.to_netcdf(output_path)
            else:
                raise Exception(
                    f"  No variables processed from {input_zip_filepath}, skipping save.")

            # Close datasets if they were opened
            if ds_accum_era:
                ds_accum_era.close()
            if ds_instant_era and ds_instant_era is not ds_accum_era:
                ds_instant_era.close()

    except zipfile.BadZipFile:
        print(
            f"Error: File {input_zip_filepath} is not a valid ZIP archive or is corrupted.")
    except Exception as e:
        print(
            f"An error occurred while processing {input_zip_filepath}: {e}")

    print("✓ ERA5 to CMIP processing complete.")
    print("\n--- Notes on Units and Calculations ---")
    print("Temperatures (tas, sst): Kept in Kelvin (K).")
    print("Pressures (pres_msl): Kept in Pascals (Pa).")
    print("Radiation fluxes (rlut, rsut, rsds): Converted from accumulated J m-2 to W m-2 (average rate over the hour).")
    print("  rlut = - (top_net_thermal_radiation / 3600)")
    print("  rsut = (toa_incident_solar_radiation - top_net_solar_radiation) / 3600")
    print("  rsds = surface_solar_radiation_downwards / 3600")
    print("Precipitation (pr): Converted from accumulated meters to kg m-2 s-1.")
    print("  pr = (total_precipitation_meters * 1000_kg_m-3) / 3600_s")
    print("Sea Ice Cover (sic): Kept as a fraction (0-1). Multiply by 100 if percentage is needed.")


def convert_to_healpix(cmip_dataset_path, output_path, hpx_level=6):
    ds = xr.open_dataset(cmip_dataset_path)
    # Convert to HEALPix using earth2grid
    print(
        f"\nStarting conversion to HEALPix CSR format (level {hpx_level})...")
    # Get the lat-lon coordinates from the dataset
    lat, lon = ds.latitude.values, ds.longitude.values
    print(f"Dataset coordinates: lat shape {lat.shape}, lon shape {lon.shape}")

    # Create source grid (lat-lon)
    src_grid = earth2grid.latlon.LatLonGrid(lat=lat, lon=lon, cylinder=False)
    # Create target HEALPix grid (CSR format)
    # Choose resolution level - higher level = higher resolution
    # Level 6 gives nside=64, level 7 gives nside=128, etc.
    target_grid = earth2grid.healpix.Grid(
        # CSR typically uses NEST ordering
        level=hpx_level,  pixel_order=earth2grid.healpix.PixelOrder.RING
    )

    print(
        f"Target HEALPix grid: level {hpx_level}, nside {2**hpx_level}, npix {target_grid.shape[0]}")

    # Create regridder from lat-lon to HEALPix
    regridder = earth2grid.get_regridder(src_grid, target_grid).float()

    print("Regridder created successfully")

    # Convert each variable to HEALPix format
    healpix_data = {}

    # Get variables that actually exist in the dataset
    existing_vars = [var for var in VARS_TO_KEEP if var in ds.data_vars]
    print(f"Converting variables: {existing_vars}")

    for var_name in existing_vars:
        print(f"Converting {var_name}...")
        var_data = ds[var_name].values

        # Handle different dimensions
        if var_data.ndim == 2:  # (lat, lon)
            # Convert to tensor and ensure float32 precision
            var_tensor = torch.from_numpy(var_data.astype(np.float32))
            var_healpix = regridder(var_tensor)
            healpix_data[var_name] = var_healpix.numpy()
        elif var_data.ndim == 3:  # (time, lat, lon)
            healpix_time_series = []
            for t in range(var_data.shape[0]):
                var_tensor = torch.from_numpy(var_data[t].astype(np.float32))
                var_healpix = regridder(var_tensor)
                healpix_time_series.append(var_healpix.numpy())
            healpix_data[var_name] = np.stack(healpix_time_series, axis=0)
        else:
            print(
                f"Skipping {var_name} - unsupported dimensions: {var_data.shape}")
            continue

        print(
            f"  {var_name}: {var_data.shape} -> {healpix_data[var_name].shape}")

    # Create new dataset with HEALPix data
    print("\nCreating HEALPix dataset...")

    # Create coordinate variables
    npix = target_grid.shape[0]
    pix_coord = np.arange(npix)

    # Create data variables for the new dataset
    data_vars = {}
    coords = {'pix': pix_coord}

    # Add time coordinate if it exists
    if 'valid_time' in ds.coords:
        coords['time'] = ds.valid_time

    # Add CRS information for HEALPix
    crs_attrs = {
        'grid_mapping_name': 'healpix',
        'healpix_nside': 2**hpx_level,
        'healpix_order': 'ring'
    }
    coords['crs'] = xr.DataArray(
        data=[np.nan],  # Dummy value
        dims=['crs'],
        attrs=crs_attrs
    )

    for var_name, var_data in healpix_data.items():
        if var_data.ndim == 1:  # Single time step
            data_vars[var_name] = (['pix'], var_data, {'grid_mapping': 'crs'})
        elif var_data.ndim == 2:  # Time series
            data_vars[var_name] = (['time', 'pix'], var_data, {
                'grid_mapping': 'crs'})

    # Create the new dataset
    ds_healpix = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=ds.attrs.copy()
    )

    # Add processing history
    if 'history' in ds_healpix.attrs:
        ds_healpix.attrs['history'] += f"\nConverted to HEALPix (level {hpx_level}, NEST ordering) using earth2grid"
    else:
        ds_healpix.attrs['history'] = f"Converted to HEALPix (level {hpx_level}, NEST ordering) using earth2grid"

    print(f"HEALPix dataset created with {len(data_vars)} variables")
    print(f"Dataset shape: {ds_healpix.dims}")

    # Save to new file
    print(f"\nSaving to: {output_path}")
    ds_healpix.to_netcdf(output_path)
    print("Conversion complete!")
    # Display the new dataset
    print("\nHEALPix dataset summary:")
    print(ds_healpix)

    # %%
    # Verify the conversion by checking some statistics
    print("\nVerification - Variable statistics:")
    for var_name in ds_healpix.data_vars:
        var = ds_healpix[var_name]
        print(
            f"{var_name}: shape={var.shape}, min={var.min().values:.4f}, max={var.max().values:.4f}")

    print(f"\nOutput file saved: {output_path}")
    print("You can now use this HEALPix dataset with the cBottle models!")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ERA5 data, process to CMIP format, and convert to HEALPix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process data for a specific date
  %(prog)s --year 2020 --month 1 --day 15 --hour "12:00"
  
  # Use custom output directory and HEALPix level
  %(prog)s --year 2020 --month 1 --day 15 --output-dir /path/to/output --healpix-level 7
  
  # Force re-download and re-processing
  %(prog)s --year 2020 --month 1 --day 15 --force
  
  # Download only specific variables
  %(prog)s --year 2020 --month 1 --day 15 --variables "2m_temperature" "total_precipitation"
        """
    )
    # Add date range options for last 10 days
    today = pd.Timestamp.now()
    ten_days_ago = today - pd.Timedelta(days=10)
    year = ten_days_ago.year
    month = ten_days_ago.month
    day = ten_days_ago.day
    hour = "12:00"
    # Date and time parameters
    parser.add_argument(
        '--year', type=int, default=year,
        help='Year to download (e.g., 2020)'
    )
    parser.add_argument(
        '--month', type=int, default=month,
        help='Month to download (1-12)'
    )
    parser.add_argument(
        '--day', type=int, default=day,
        help='Day to download (1-31)'
    )
    parser.add_argument(
        '--hour', type=str, default=hour,
        help='Hour to download in HH:MM format (default: "12:00")'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for all files (default: $XDG_CACHE_HOME/cbottle or ~/.cache/cbottle)'
    )

    # Processing options
    parser.add_argument(
        '--healpix-level', type=int, default=6,
        help='HEALPix resolution level (default: 6, nside=64)'
    )
    parser.add_argument(
        '--variables', nargs='*', default=None,
        help='Specific ERA5 variables to download (default: all standard variables)'
    )

    # Control options
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download and re-processing even if files exist'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip download step (assumes download file already exists)'
    )
    parser.add_argument(
        '--skip-cmip', action='store_true',
        help='Skip CMIP processing step (assumes CMIP file already exists)'
    )
    parser.add_argument(
        '--skip-healpix', action='store_true',
        help='Skip HEALPix conversion step'
    )
    return parser.parse_args()


def process_era5_data(args):
    """
    Process ERA5 data through the complete pipeline: download, CMIP conversion, and HEALPix transformation.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing:
        - year, month, day: Date parameters
        - hour: Hour in HH:MM format
        - output_dir: Output directory path
        - healpix_level: HEALPix resolution level
        - variables: List of ERA5 variables to download
        - force: Whether to force re-download and re-processing
        - skip_download: Whether to skip download step
        - skip_cmip: Whether to skip CMIP processing step
        - skip_healpix: Whether to skip HEALPix conversion step
    
    Returns:
    --------
    str : Path to the final HEALPix output file
    """
    # Validate date parameters
    if not (1 <= args.month <= 12):
        raise ValueError("Month must be between 1 and 12")
    if not (1 <= args.day <= 31):
        raise ValueError("Day must be between 1 and 31")
    
    # Validate hour format
    try:
        hour_parts = args.hour.split(':')
        if len(hour_parts) != 2 or not (0 <= int(hour_parts[0]) <= 23) or not (0 <= int(hour_parts[1]) <= 59):
            raise ValueError
    except ValueError:
        raise ValueError("Hour must be in HH:MM format (e.g., '12:00')")

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.expanduser("~"), ".cache")
        output_dir = os.path.join(output_dir, "cbottle")
    print(f"Using output directory: {output_dir}")
    
    # Create directory structure
    era_dir_path = os.path.join(output_dir, "era5_data")
    cmip_dir_path = os.path.join(output_dir, "cmip_processed")
    healpix_dir_path = os.path.join(output_dir, "healpix_processed")

    os.makedirs(era_dir_path, exist_ok=True)
    os.makedirs(cmip_dir_path, exist_ok=True)
    os.makedirs(healpix_dir_path, exist_ok=True)

    # Generate file paths
    download_file = os.path.join(
        era_dir_path, f"era5_variables_{args.year}{args.month:02d}{args.day:02d}{args.hour[:2]}.zip")

    filename = os.path.basename(download_file).replace(
        "era5_variables", "era5_cmip_processed").replace(".zip", ".nc")
    cmip_output_filepath = os.path.join(cmip_dir_path, filename)

    healpix_output_filepath = os.path.join(
        healpix_dir_path, f"era5_data_cmip_healpix_{args.year}{args.month:02d}{args.day:02d}{args.hour[:2]}.nc")
    
    print(f"Processing date: {args.year}-{args.month:02d}-{args.day:02d} {args.hour}")
    print(f"Files will be saved as:")
    print(f"  Download: {download_file}")
    print(f"  CMIP: {cmip_output_filepath}")
    print(f"  HEALPix: {healpix_output_filepath}")

    # Step 1: Download ERA5 data
    if not args.skip_download:
        if not os.path.exists(download_file) or args.force:
            try:
                download_era5_data(
                    year=args.year,
                    month=args.month,
                    day=args.day,
                    hour=args.hour,
                    output_path=download_file,
                    variables=args.variables
                )
            except Exception as e:
                raise Exception(f"Download failed: {e}")
        else:
            print(f"Download file already exists: {download_file}")
    else:
        print("Skipping download step")
        if not os.path.exists(download_file):
            raise FileNotFoundError(f"Download file does not exist: {download_file}")

    # Step 2: Process to CMIP format
    if not args.skip_cmip:
        if not os.path.exists(cmip_output_filepath) or args.force:
            try:
                process_era5_to_cmip(
                    input_download_file=download_file,
                    output_path=cmip_output_filepath
                )
            except Exception as e:
                raise Exception(f"CMIP processing failed: {e}")
        else:
            print(f"CMIP file already exists: {cmip_output_filepath}")
    else:
        print("Skipping CMIP processing step")
        if not os.path.exists(cmip_output_filepath):
            raise FileNotFoundError(f"CMIP file does not exist: {cmip_output_filepath}")

    # Step 3: Convert to HEALPix
    if not args.skip_healpix:
        if not os.path.exists(healpix_output_filepath) or args.force:
            try:
                convert_to_healpix(
                    cmip_dataset_path=cmip_output_filepath,
                    output_path=healpix_output_filepath,
                    hpx_level=args.healpix_level
                )
            except Exception as e:
                raise Exception(f"HEALPix conversion failed: {e}")
        else:
            print(f"HEALPix file already exists: {healpix_output_filepath}")
    else:
        print("Skipping HEALPix conversion step")

    print(f"\n🎉 Processing complete!")
    print(f"Final output: {healpix_output_filepath}")
    
    return healpix_output_filepath


if __name__ == "__main__":
    args = parse_arguments()
    try:
        process_era5_data(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)