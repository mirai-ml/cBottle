#!/usr/bin/env python3
from era5_dataprocessing import process_era5_data
from argparse import Namespace
from datetime import datetime, timedelta
import os
import time

try:
    from simple_slurm import Slurm
except ImportError:
    print("simple_slurm is not installed. Please install it with: pip install simple_slurm")
    exit(1)

def date_range(start_date, end_date):
    """Generate dates between start_date and end_date inclusive"""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def submit_era5_job(date, output_dir):
    """Submit a SLURM job for processing a single date using simple_slurm"""
    
    # Create the Slurm object with job configuration
    slurm = Slurm(
        job_name=f"era5_{date.strftime('%Y%m%d')}",
        partition="normal",
        time="00:20:00",
        mem="16G",
        cpus_per_task=4,
        output=f"logs/era5_{date.strftime('%Y%m%d')}_%j.out",
        error=f"logs/era5_{date.strftime('%Y%m%d')}_%j.err"
    )
    
    # Create the Python command to run
    python_command = f"""
python -c '
from era5_dataprocessing import process_era5_data
from argparse import Namespace

args = Namespace(
    year={date.year},
    month={date.month},
    day={date.day},
    hour="12:00",
    output_dir="{output_dir}",
    healpix_level=6,
    variables=None,
    force=False,
    skip_download=True,
    skip_cmip=True,
    skip_healpix=False
)

process_era5_data(args)
'
"""
    
    # Submit the job
    try:
        job_id = slurm.sbatch(python_command.strip())
        print(f"Submitted job for {date.strftime('%Y-%m-%d')}: Job ID {job_id}")
        return job_id
    except Exception as e:
        print(f"Failed to submit job for {date.strftime('%Y-%m-%d')}: {e}")
        return None

def main():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate dates from May 2024 to May 2025
    start = datetime(2024, 5, 1)
    end = datetime(2025, 5, 31)
    output_dir = "/scratch/mirai/experiments/062025_healpix_1year"
    
    # Sleep duration between job submissions (in seconds)
    sleep_duration = 10  # Adjust this value as needed

    # Submit a job for each date
    job_ids = []
    total_dates = len(list(date_range(start, end)))
    
    count = 0
    for i, date in enumerate(date_range(start, end)):
        print(f"Submitting job {i+1}/{total_dates} for {date.strftime('%Y-%m-%d')}...")
        
        filepath = f"{output_dir}/healpix_processed/era5_data_cmip_healpix_{date.year}{date.month:02d}{date.day:02d}12.nc"
        if os.path.exists(filepath):
            print(f"Skipping {date.strftime('%Y-%m-%d')} because it already exists")
            continue
        job_id = submit_era5_job(date, output_dir)
        if job_id:
            job_ids.append(job_id)
        
        # Sleep between submissions (except for the last one)
        # if i < total_dates - 1:
        #     print(f"Waiting {sleep_duration} seconds before next submission...")
        #     time.sleep(sleep_duration)
        # count+=1
        # if count > 50:
        #     time.sleep(600)
            # break
    
    print(f"\nSubmitted {len(job_ids)} jobs successfully")
    if job_ids:
        print(f"Job IDs: {', '.join(map(str, job_ids))}")

if __name__ == "__main__":
    main()





# %%
