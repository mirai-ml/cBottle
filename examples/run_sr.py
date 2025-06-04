#%%
import sys
import xarray as xr
sys.path.append("../scripts")
from inference_multidiffusion import inference as inference_super_resolution

# Register blosc2 codec to fix UnknownCodecError
try:
    import blosc2
    import numcodecs
    from numcodecs.abc import Codec
    
    class Blosc2(Codec):
        codec_id = 'blosc2'
        
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        def encode(self, buf):
            return blosc2.compress(buf, **self.kwargs)
        
        def decode(self, buf, out=None):
            return blosc2.decompress(buf)
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    # Register the codec if not already registered
    if 'blosc2' not in numcodecs.registry.codec_registry:
        numcodecs.register_codec(Blosc2)
except ImportError:
    print("Warning: blosc2 not available, some zarr files may not be readable")

import os
import cbottle.config.environment as config
import earth2grid
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from cbottle import checkpointing, patchify, visualizations
from cbottle.datasets import samplers
from cbottle.datasets.dataset_2d import HealpixDatasetV5, NetCDFWrapperV1
from cbottle.diffusion_samplers import edm_sampler
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from earth2grid import healpix
import click
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import tqdm
import xarray as xr

def diagnostics(pred, lr, target):
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
        plt.savefig(f"output_{var}")

@click.command()
@click.argument('--checkpoint', type=str)
@click.argument('output_path', type=str)
@click.option('--input-path', type=str, default="", help="Path to the input data")
@click.option('--plot-sample', is_flag=True, help="Plot samples")
@click.option('--min-samples', type=int, default=1, help="Number of samples to inference")
@click.option('--level', type=int, default=10, help="HPX level for high res input")
@click.option('--level-lr', type=int, default=6, help="HPX level for low res input")
@click.option('--patch-size', type=int, default=128, help="Patch size for multidiffusion")
@click.option('--overlap-size', type=int, default=32, help="Overlapping pixel number between patches")
@click.option('--num-steps', type=int, default=18, help="Sampler iteration number")
@click.option('--sigma-max', type=int, default=800, help="Noise sigma max")
@click.option('--super-resolution-box', type=(int, int, int, int), default=None, 
              help="Bounding box (lat_south lon_west lat_north lon_east) where super-resolution will be applied. Regions outside the box remain coarse.")
def inference(checkpoint, output_path, input_path, plot_sample, min_samples, level, level_lr, 
              patch_size, overlap_size, num_steps, sigma_max, super_resolution_box, customized_dataset=None):
    hpx_level = level
    hpx_lr_level = level_lr
    box = super_resolution_box

    LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    WORLD_RANK = int(os.getenv("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")

    dist.init_process_group(backend="nccl", init_method="env://", world_size=WORLD_SIZE, rank=WORLD_RANK)
    torch.cuda.set_device(LOCAL_RANK)

    if torch.cuda.is_available():
        if LOCAL_RANK is not None:
            device = torch.device(f"cuda:{LOCAL_RANK}")
        else:
            device = torch.device("cuda")
    if customized_dataset is not None:
        test_dataset = customized_dataset(
            split="test",
        )
        tasks = None
    elif input_path:
        ds = xr.open_dataset(input_path)
        test_dataset = NetCDFWrapperV1(ds, hpx_level=hpx_level, healpixpad_order=False)
        tasks = None
    else:
        test_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            land_path=config.LAND_DATA_URL_10,
            train=False,
            yield_index=True,
            healpixpad_order=False,
        )
        sampler = samplers.subsample(test_dataset, min_samples=min_samples)
        tasks = samplers.distributed_split(sampler)

    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, sampler=tasks
    )

    # Initialize netCDF writer
    nc_config = NetCDFConfig(
        hpx_level=hpx_level,
        time_units=test_dataset.time_units,
        calendar=test_dataset.calendar,
        attrs={},
    )
    writer = NetCDFWriter(
        output_path, nc_config, test_dataset.batch_info.channels, rank=WORLD_RANK
    )

    in_channels = len(test_dataset.fields_out)

    with checkpointing.Checkpoint(checkpoint) as checkpoint:
        net = checkpoint.read_model()

    net.eval().requires_grad_(False).cuda()

    torch.cuda.set_device(LOCAL_RANK)
    net.cuda(LOCAL_RANK)

    # setup grids
    high_res_grid = healpix.Grid(level=hpx_level, pixel_order=healpix.PixelOrder.NEST)
    low_res_grid = healpix.Grid(level=hpx_lr_level, pixel_order=healpix.PixelOrder.NEST)
    lat = torch.linspace(-90, 90, 128)[:, None]
    lon = torch.linspace(0, 360, 128)[None, :]
    regrid_to_latlon = low_res_grid.get_bilinear_regridder_to(lat, lon).cuda()
    regrid = earth2grid.get_regridder(low_res_grid, high_res_grid)
    regrid.cuda().float()

    inbox_patch_index = None
    if box is not None:
        inbox_patch_index = patchify.patch_index_from_bounding_box(
            hpx_level, box, patch_size, overlap_size, device
        )
        print(
            f"Performing super-resolution within the minimal region covering (lat_south, lon_west, lat_north, lon_east): {box} with {len(inbox_patch_index)} patches. "
        )
    else:
        print("Performing super-resolution over the entire globe")

    for batch in tqdm.tqdm(loader):
        target = batch["target"]
        target = target[0, :, 0]
        # normalize inputs
        with torch.no_grad():
            # coarsen the target map as condition if icon_v5 is used
            if not input_path:
                lr = target
                for _ in range(high_res_grid.level - low_res_grid.level):
                    npix = lr.size(-1)
                    shape = lr.shape[:-1]
                    lr = lr.view(shape + (npix // 4, 4)).mean(-1)
                inp = lr.cuda()
            else:
                inp = batch["condition"]
                inp = inp[0, :, 0]
                inp = inp.cuda(non_blocking=True)
            # get global low res
            global_lr = regrid_to_latlon(inp.double())[None,].cuda()
            lr = regrid(inp)
        latents = torch.randn_like(lr)
        latents = latents.reshape((in_channels, -1))
        lr = lr.reshape((in_channels, -1))
        target = target.reshape((in_channels, -1))
        latents = latents[None,].to(device)
        lr = lr[None,].to(device)
        target = target[None,].to(device)
        with torch.no_grad():
            # scope with global_lr and other inputs present
            def denoiser(x, t):
                return (
                    patchify.apply_on_patches(
                        net,
                        patch_size=patch_size,
                        overlap_size=overlap_size,
                        x_hat=x,
                        x_lr=lr,
                        t_hat=t,
                        class_labels=None,
                        batch_size=128,
                        global_lr=global_lr,
                        inbox_patch_index=inbox_patch_index,
                        device=device,
                    )
                    .to(torch.float64)
                    .cuda()
                )

            denoiser.sigma_max = net.sigma_max
            denoiser.sigma_min = net.sigma_min
            denoiser.round_sigma = net.round_sigma
            pred = edm_sampler(
                denoiser,
                latents,
                num_steps=num_steps,
                sigma_max=sigma_max,
            )
        pred = pred.cpu() * test_dataset._scale + test_dataset._mean
        lr = lr.cpu() * test_dataset._scale + test_dataset._mean
        target = target.cpu() * test_dataset._scale + test_dataset._mean

        def prepare(x):
            ring_order = high_res_grid.reorder(earth2grid.healpix.PixelOrder.RING, x)
            return {
                test_dataset.batch_info.channels[c]: ring_order[:, c, None].cpu()
                for c in range(x.shape[1])
            }

        output_data = prepare(pred)
        # Convert time data to timestamps
        timestamps = batch["timestamp"]
        writer.write_batch(output_data, timestamps)

    if WORLD_RANK == 0 and plot_sample:
        input_data = prepare(lr)
        target_data = prepare(target)
        diagnostics(
            output_data,
            input_data,
            target_data,
        )

#%%
vars_to_keep = [
    'cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 
    'rsut', 'pres_msl', 'pr', 'rsds', 'sst', 'sic','crs'
]

#%%
data = xr.open_zarr("~/mirai_scratch/20250510_00.zarr")
# %%
data.data_vars.keys()
# %%
