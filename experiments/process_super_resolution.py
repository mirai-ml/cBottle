#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import argparse
from pathlib import Path
import time

import cbottle.config.environment as config
import earth2grid
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from cbottle import checkpointing, patchify, visualizations
from cbottle.datasets.dataset_2d import NetCDFWrapperV1
from cbottle.diffusion_samplers import edm_sampler
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from earth2grid import healpix

import numpy as np
import tqdm
import xarray as xr

try:
    from simple_slurm import Slurm
except ImportError:
    print("simple_slurm is not installed. Please install it with: pip install simple_slurm")
    exit(1)


def diagnostics(pred, lr, target, output_dir, filename_prefix): """Generate diagnostic plots comparing input, prediction, and target"""
    titles = ["input", "prediction", "target"]
    for var in pred.keys():
        plt.figure(figsize=(50, 25))
        vmin = torch.min(pred[var][0, 0])
        vmax = torch.max(pred[var][0, 0])
        for idx, data, title in zip(
            np.arange(1, 4), [lr[var][0, 0], pred[var][0, 0], target[var][0, 0]], titles
        ):
            visualizations.visualize(
                data,
                pos=(1, 3, idx),
                title=title,
                nlat=1024,
                nlon=2048,
                vmin=vmin,
                vmax=vmax,
            )
        plt.tight_layout()
        
        # Save figure in the specified diagnostics folder
        figure_path = os.path.join(output_dir, f"{filename_prefix}_{var}.png")
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"Saved diagnostic plot: {figure_path}")


def prepare_output_data(x, high_res_grid, dataset):
    """Prepare output data for writing"""
    ring_order = high_res_grid.reorder(earth2grid.healpix.PixelOrder.RING, x)
    return {
        dataset.batch_info.channels[c]: ring_order[:, c, None].cpu()
        for c in range(x.shape[1])
    }


def process_single_file(input_file, state_path, output_dir, args):
    """Process a single healpix file through super resolution"""
    
    # Setup distributed processing
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    WORLD_RANK = int(os.getenv("RANK", 0))
    
    if WORLD_SIZE > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12345")
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=WORLD_SIZE, rank=WORLD_RANK
        )
        torch.cuda.set_device(LOCAL_RANK)

    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

    # Load dataset from file
    ds = xr.open_dataset(input_file)
    dataset = NetCDFWrapperV1(ds, hpx_level=args.level, healpixpad_order=False)
    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

    # Setup output path
    input_filename = Path(input_file).stem
    output_file = os.path.join(output_dir, f"{input_filename}_sr.nc")

    # Initialize netCDF writer
    nc_config = NetCDFConfig(
        hpx_level=args.level,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        attrs={},
    )
    writer = NetCDFWriter(
        output_file, nc_config, dataset.batch_info.channels, rank=WORLD_RANK
    )

    in_channels = len(dataset.fields_out)

    # Load model
    with checkpointing.Checkpoint(state_path) as checkpoint:
        net = checkpoint.read_model()

    net.eval().requires_grad_(False).to(device)

    # Setup grids
    high_res_grid = healpix.Grid(level=args.level, pixel_order=healpix.PixelOrder.NEST)
    low_res_grid = healpix.Grid(level=args.level_lr, pixel_order=healpix.PixelOrder.NEST)
    lat = torch.linspace(-90, 90, 128)[:, None]
    lon = torch.linspace(0, 360, 128)[None, :]
    regrid_to_latlon = low_res_grid.get_bilinear_regridder_to(lat, lon).to(device)
    regrid = earth2grid.get_regridder(low_res_grid, high_res_grid)
    regrid.to(device).float()

    # Setup patch index for bounding box if provided
    inbox_patch_index = None
    if args.super_resolution_box:
        box = tuple(args.super_resolution_box)
        inbox_patch_index = patchify.patch_index_from_bounding_box(
            args.level, box, args.patch_size, args.overlap_size, device
        )
        print(f"Processing super-resolution within bounding box: {box} with {len(inbox_patch_index)} patches")
    else:
        print("Processing super-resolution over the entire globe")

    # Process batches
    for batch in tqdm.tqdm(loader, desc=f"Processing {input_filename}"):
        target = batch["target"]
        target = target[0, :, 0]
        
        with torch.no_grad():
            # Get input data
            inp = batch["condition"]
            inp = inp[0, :, 0]
            inp = inp.to(device, non_blocking=True)
            
            # Get global low res
            global_lr = regrid_to_latlon(inp.double())[None,].to(device)
            lr = regrid(inp)
            
        latents = torch.randn_like(lr)
        latents = latents.reshape((in_channels, -1))
        lr = lr.reshape((in_channels, -1))
        target = target.reshape((in_channels, -1))
        latents = latents[None,].to(device)
        lr = lr[None,].to(device)
        target = target[None,].to(device)
        
        with torch.no_grad():
            def denoiser(x, t):
                return (
                    patchify.apply_on_patches(
                        net,
                        patch_size=args.patch_size,
                        overlap_size=args.overlap_size,
                        x_hat=x,
                        x_lr=lr,
                        t_hat=t,
                        class_labels=None,
                        batch_size=args.batch_size,
                        global_lr=global_lr,
                        inbox_patch_index=inbox_patch_index,
                        device=device,
                    )
                    .to(torch.float64)
                    .to(device)
                )

            denoiser.sigma_max = net.sigma_max
            denoiser.sigma_min = net.sigma_min
            denoiser.round_sigma = net.round_sigma
            
            pred = edm_sampler(
                denoiser,
                latents,
                num_steps=args.num_steps,
                sigma_max=args.sigma_max,
            )
        
        pred = pred.cpu() * dataset._scale + dataset._mean
        lr = lr.cpu() * dataset._scale + dataset._mean
        target = target.cpu() * dataset._scale + dataset._mean

        output_data = prepare_output_data(pred, high_res_grid, dataset)
        timestamps = batch["timestamp"]
        writer.write_batch(output_data, timestamps)
        
        # Generate diagnostics if requested
        if args.generate_diagnostics:
            diagnostics_dir = os.path.join(output_dir, "diagnostics")
            os.makedirs(diagnostics_dir, exist_ok=True)
            
            input_data = prepare_output_data(lr, high_res_grid, dataset)
            target_data = prepare_output_data(target, high_res_grid, dataset)
            
            # Create filename prefix from input file and timestamp
            timestamp_str = str(timestamps[0].values).replace(':', '-').replace(' ', '_')
            filename_prefix = f"{input_filename}_{timestamp_str}"
            
            diagnostics(
                output_data,
                input_data, 
                target_data,
                diagnostics_dir,
                filename_prefix
            )

    print(f"Completed processing {input_filename} -> {output_file}")


def submit_sr_job(input_file, state_path, output_dir, args):
    """Submit a SLURM job for super resolution processing of a single file"""
    
    input_filename = Path(input_file).stem
    
    # Create the Slurm object with job configuration
    slurm = Slurm(
        job_name=f"sr_{input_filename}",
        partition=args.partition,
        time=args.time_limit,
        mem=args.memory,
        cpus_per_task=args.cpus_per_task,
        gres=f"gpu:{args.gpus_per_node}",
        output=f"logs/sr_{input_filename}_%j.out",
        error=f"logs/sr_{input_filename}_%j.err",
        exclude=f"margpu018,margpu002,margpu003"
    )
    
    # Create the Python command to run
    python_command = f"""
echo $HOSTNAME
CUDA_VISIBLE_DEVICES=0 python -c '
import sys
sys.path.append("{os.getcwd()}")
from process_super_resolution import process_single_file
import argparse

args = argparse.Namespace(level={args.level}, level_lr={args.level_lr}, patch_size={args.patch_size}, overlap_size={args.overlap_size}, num_steps={args.num_steps}, sigma_max={args.sigma_max}, batch_size={args.batch_size}, super_resolution_box={args.super_resolution_box}, generate_diagnostics={args.generate_diagnostics})
process_single_file("{input_file}", "{state_path}", "{output_dir}", args)
'
"""
    
    # Submit the job
    try:
        job_id = slurm.sbatch(python_command.strip())
        print(f"Submitted super-resolution job for {input_filename}: Job ID {job_id}")
        return job_id
    except Exception as e:
        print(f"Failed to submit job for {input_filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Super Resolution Processing with CBOTTLE")
    parser.add_argument("state_path", type=str, help="Path to the model state file")
    parser.add_argument("--root-path", type=str, 
                       default="/scratch/mirai/experiments/062025_healpix_1year/healpix_processed",
                       help="Root path containing healpix processed files")
    parser.add_argument("--output-dir", type=str, 
                       default="/scratch/mirai/experiments/062025_healpix_1year/super_resolution",
                       help="Output directory for super resolution results")
    parser.add_argument("--pattern", type=str, default="era5_data_cmip_healpix_*.nc",
                       help="File pattern to match in root path")
    
    # Model and processing parameters
    parser.add_argument("--level", type=int, default=10, help="HPX level for high res output")
    parser.add_argument("--level-lr", type=int, default=6, help="HPX level for low res input")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size for multidiffusion")
    parser.add_argument("--overlap-size", type=int, default=32, help="Overlapping pixel number between patches")
    parser.add_argument("--num-steps", type=int, default=18, help="Sampler iteration number")
    parser.add_argument("--sigma-max", type=int, default=800, help="Noise sigma max")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for patches")
    parser.add_argument("--super-resolution-box", type=int, nargs=4, default=None,
                       metavar=("lat_south", "lon_west", "lat_north", "lon_east"),
                       help="Bounding box for super-resolution region")
    parser.add_argument("--generate-diagnostics", action="store_true", 
                       help="Generate diagnostic plots comparing input, prediction, and target")
    
    # SLURM parameters
    parser.add_argument("--submit-jobs", action="store_true", help="Submit SLURM jobs instead of running locally")
    parser.add_argument("--partition", type=str, default="gpu-best,gpu", help="SLURM partition")
    parser.add_argument("--time-limit", type=str, default="02:00:00", help="Time limit per job")
    parser.add_argument("--memory", type=str, default="32G", help="Memory per job")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="CPUs per task")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs per node")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--sleep-duration", type=int, default=2, help="Sleep between job submissions")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Find all healpix files
    input_files = glob.glob(os.path.join(args.root_path, args.pattern))
    input_files.sort()
    
    if args.max_files:
        input_files = input_files[:args.max_files]
    
    print(f"Found {len(input_files)} files to process")
    
    if not input_files:
        print(f"No files found matching pattern {args.pattern} in {args.root_path}")
        return
    
    if args.submit_jobs:
        # Submit SLURM jobs
        job_ids = []
        for i, input_file in enumerate(input_files):
            output_file = os.path.join(args.output_dir, f"{Path(input_file).stem}_sr.nc")
            
            # Skip if output already exists
            if os.path.exists(output_file):
                print(f"Skipping {Path(input_file).name} - output already exists")
                continue
            
            print(f"Submitting job {i+1}/{len(input_files)} for {Path(input_file).name}...")
            job_id = submit_sr_job(input_file, args.state_path, args.output_dir, args)
            if job_id:
                job_ids.append(job_id)
            
            # Sleep between submissions
            if i < len(input_files) - 1:
                print(f"Waiting {args.sleep_duration} seconds before next submission...")
                time.sleep(args.sleep_duration)
        
        print(f"\nSubmitted {len(job_ids)} super-resolution jobs successfully")
        if job_ids:
            print(f"Job IDs: {', '.join(map(str, job_ids))}")
    else:
        # Process files locally
        for i, input_file in enumerate(input_files):
            output_file = os.path.join(args.output_dir, f"{Path(input_file).stem}_sr.nc")
            
            # Skip if output already exists
            if os.path.exists(output_file):
                print(f"Skipping {Path(input_file).name} - output already exists")
                continue
            
            print(f"Processing file {i+1}/{len(input_files)}: {Path(input_file).name}")
            process_single_file(input_file, args.state_path, args.output_dir, args)


if __name__ == "__main__":
    main()
