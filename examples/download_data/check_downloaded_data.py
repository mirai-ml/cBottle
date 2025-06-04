# %%
import xarray as xr


vars_to_keep = ['cllvi', 'clivi', 'tas', 'uas', 'vas', 'rlut', 
    'rsut', 'pres_msl', 'pr', 'rsds', 'sst', 'sic']

path_raw = "/home/stip-sac/nbelkhir/workspace/mirai/xp/downscaling/cBottle/examples/download_data/data/temp/*.nc"
path_processed = "/home/stip-sac/nbelkhir/workspace/mirai/xp/downscaling/cBottle/examples/download_data/data/processed_era5_cmip/era5_cmip_processed_20220101.nc"

#%%
ds = xr.open_dataset(path_processed)
for var in vars_to_keep:
    print(var,':', var in ds.data_vars)
# %%

ds = xr.open_mfdataset(path_raw)

ds
# %%
# %%

import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["sea_ice_cover"],
    "year": ["2025"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "zip"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
# %%

ds = xr.open_dataset("/home/stip-sac/nbelkhir/workspace/mirai/xp/downscaling/cBottle/examples/download_data/data.grib", engine="cfgrib")
# %%
ds
# %%
