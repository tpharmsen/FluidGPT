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
    def __init__(self, preproc_path, temporal_bundling = 1, noisetype='puregaussian', from_frame = 5, fulltrajmode=False, **kwargs):
        self.filepath = preproc_path
        self.from_frame = from_frame
        self.noisetype = noisetype
        self._file = None
        self.fulltrajmode = fulltrajmode

        if self.noisetype == "gaussiangaussian":
            self.sigma_time = kwargs.get("sigma_time", 1.0)
            self.sigma_space = kwargs.get("sigma_space", 1.0)

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
        self.lenpertraj = self.ts - self.dt * self.tb + self.dt #self.ts - self.dt * self.tb
        self.idx_window = self.dt * self.tb
        self.avgnorm = None
        self.stdnorm = None
        #print(self.ts, self.idx_window, self.lenpertraj, self.dt, self.tb)
        
    def __len__(self):
        return self.traj * self.lenpertraj
        #return 1
            
    def __getitem__(self, idx):
        if not self.fulltrajmode:
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
            """
            if self.noisetype == 'puregaussian':
                prior = self.prior_purenoise(target.clone(), fromframe=self.from_frame)
            elif self.noisetype == 'checkerboard':
                prior = self.checkerboard_noise(target.clone(), fromframe=self.from_frame)
            elif self.noisetype == 'avggaussian':
                prior = self.prior_avggaussian(target.clone(), fromframe=self.from_frame, smooth_passes=3)
            elif self.noisetype == 'gaussiangaussian':
                #target_clone = target.clone().permute(1,0,2,3)
                prior = self.prior_gaussiangaussian(target.clone(), fromframe=self.from_frame, sigma_time=self.sigma_time, sigma_space=self.sigma_space)
                #prior = prior.permute(1,0,2,3)
            else:
                raise ValueError(f"Unknown noisetype {self.noisetype}")
            #label = (label - self.avgnorm) / self.stdnorm)
            """
            return torch.tensor(target, dtype=torch.float32).permute(1,0,2,3)
        else:
            return self.get_single_traj(idx)

    def get_single_traj(self, idx):
        #f = self._get_file()
        filename = self.filepath
        filename = Path(filename)
        filename = filename / f'traj{idx:05d}.h5'
        with h5py.File(filename, 'r') as f:
            full = torch.from_numpy(f['data'][::self.dt])
        if self.avgnorm is not None:
            full = (full - self.avgnorm) / self.stdnorm

        return torch.tensor(full, dtype=torch.float32).permute(1,0,2,3).unsqueeze(0)