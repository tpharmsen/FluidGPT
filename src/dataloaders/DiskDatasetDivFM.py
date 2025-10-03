import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from dataloaders.utils import spatial_resample

from scipy.ndimage import gaussian_filter

class DiskDatasetDivFM(Dataset):
    def __init__(self, preproc_path, temporal_bundling = 1, noisetype='puregaussian', from_frame = 4):
        self.filepath = preproc_path
        self.from_frame = from_frame
        self.noisetype = noisetype
        self._file = None
        # If the file ends with .h5, remove it
        if self.filepath.endswith('.h5'):
            self.filepath = self.filepath[:-3]
        metafile = Path(self.filepath)
        metafile = metafile / 'meta.h5'
        #check if metafile exists
        if not metafile.exists():
            raise FileNotFoundError(f"Metadata file {metafile} does not exist. Please preprocess the data first.")
        with h5py.File(metafile, 'r') as f:
            self.avg = float(f['avg'][()])
            self.std = float(f['std'][()])
            self.resample_shape = int(f['resample_shape'][()])
            self.resample_mode = str(f['resample_mode'][()].decode('utf-8'))
            self.dt = int(f['timesample'][()])
            self.name = str(f['name'][()].decode('utf-8'))
            self.traj = int(f['traj'][()])
            self.ts = int(f['ts'][()])
            self.datashape = tuple(f['datashape'][()])
        #print(f"Dataset {self.name} loaded with {self.traj} trajectories, each with {self.ts} time steps.")
        #print(f"reshape method: {self.resample_mode}, shape: {self.resample_shape}")
        self.tb = temporal_bundling
        self.lenpertraj = self.ts - self.dt * self.tb
        self.idx_window = self.dt * self.tb
        self.avgnorm = None
        self.stdnorm = None
        #print(self.ts, self.idx_window, self.lenpertraj, self.dt, self.tb)
        
    def __len__(self):
        return self.traj * self.lenpertraj
        #return 1
            

    def __getitem__(self, idx):
        
        #f = self._get_file()
        traj_idx = idx // self.lenpertraj
        ts_idx = idx % self.lenpertraj
        filename = self.filepath
        # if ends with .h5, remove it
        filename = Path(filename)
        filename = filename / f'traj{traj_idx:05d}.h5'
        with h5py.File(filename, 'r') as f:
            target = torch.from_numpy(f['data'][ts_idx : ts_idx + self.idx_window : self.dt])
            #label = f['data'][ts_idx + self.fs * self.idx_window : ts_idx + (self.fs + 1) * self.idx_window : self.dt]
        if self.avgnorm is not None:
            #print('normalising\n')
            target = (target - self.avgnorm) / self.stdnorm
        #print(target.shape)
        #prior = self.prior_purenoise(target, fromframe=self.from_frame)
        #print(self.from_frame)
        if self.noisetype == 'puregaussian':
            prior = self.prior_purenoise(target.clone(), fromframe=self.from_frame)
        elif self.noisetype == 'checkerboard':
            prior = self.checkerboard_noise(target.clone(), fromframe=self.from_frame)
        elif self.noisetype == 'smoothgaussian':
            prior = self.prior_smoothnoise(target.clone(), fromframe=self.from_frame, smooth_passes=3)
        else:
            raise ValueError(f"Unknown noisetype {self.noisetype}")
        #label = (label - self.avgnorm) / self.stdnorm)
        return (
            torch.tensor(prior, dtype=torch.float32).permute(1,0,2,3), 
            torch.tensor(target, dtype=torch.float32).permute(1,0,2,3)
        )

    def prior_purenoise(self, data, fromframe=4):
        # generate pure gaussian noise
        noise = torch.randn(size=data[fromframe:].shape)
        data[fromframe:] = noise
        return data

    def checkerboard_noise(self, data, fromframe=4):
        # generate checkerboard noise
        noise = torch.randn(size=(data.shape[0] - fromframe, data.shape[1], data.shape[2] // 8, data.shape[3] // 8), device=data.device)
        noise = noise.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        data[fromframe:] = noise
        return data
    
    def prior_smoothnoise(self, data, fromframe=4, smooth_passes=3):
        noise = torch.randn((data.shape[0] - fromframe, data.shape[1], data.shape[2], data.shape[3]), device=data.device, dtype=data.dtype)
        def circular_avg_pool3d(x, kernel_size=(5,5,5), stride=1, passes=1):
            pad_t, pad_h, pad_w = kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t, pad_t), mode="circular")
            x = F.avg_pool3d(x, kernel_size=kernel_size, stride=stride)
            return x
        noise = noise.permute(1,0,2,3)
        for _ in range(smooth_passes):
            #print(noise.shape)
            noise = circular_avg_pool3d(
                noise,
                kernel_size=(5, 5, 5),
                stride=1,
            )
        noise = noise.permute(1,0,2,3)
        noise = noise / noise.std()  # Normalize to std=1
        data[fromframe:] = noise
        return data

    def get_single_traj(self, idx):
        #f = self._get_file()
        filename = self.filepath
        filename = Path(filename)
        filename = filename / f'traj{idx:05d}.h5'
        with h5py.File(filename, 'r') as f:
            full = torch.from_numpy(f['data'][::self.dt])
        if self.avgnorm is not None:
            full = (full - self.avgnorm) / self.stdnorm
        #print(full.shape)
        #prior = self.prior_purenoise(full[:self.tb], fromframe=self.from_frame)
        if self.noisetype == 'puregaussian':
            prior = self.prior_purenoise(full[:self.tb].clone(), fromframe=self.from_frame)
        elif self.noisetype == 'checkerboard':
            prior = self.checkerboard_noise(full[:self.tb].clone(), fromframe=self.from_frame)
        elif self.noisetype == 'smoothgaussian':
            prior = self.prior_smoothnoise(full[:self.tb].clone(), fromframe=self.from_frame, smooth_passes=3)
        else:
            raise ValueError(f"Unknown noisetype {self.noisetype}")
        return (
            torch.tensor(prior, dtype=torch.float32).permute(1,0,2,3),
            torch.tensor(full, dtype=torch.float32).permute(1,0,2,3).unsqueeze(0)
        )