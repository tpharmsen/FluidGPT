# import libraries
import sys

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, ConstantLR
from datetime import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import time
import wandb
import yaml
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.animation as animation
import os
import subprocess
import platform
import json
import argparse
import warnings

from matplotlib import rcParams

rcParams.update({
    'font.size': 14,
    'figure.figsize': (8, 6),
    'axes.titlesize': 20,
    'axes.labelsize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',

    # --- Lines & markers ---
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1,
    'lines.markerfacecolor': 'none',

    # Remove top/right spines globally
    'axes.spines.top': True,
    'axes.spines.right': True,

    "legend.frameon": True,
})

sys.path.append('src/')

from dataloaders import *
from dataloaders import PREPROC_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSamplerReduced, spatial_resample
#from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout
from trainers.utils import animate_rollout, magnitude_vel, rollout_det, compute_energy_enstrophy_spectra
from trainers.utils import prior_purenoise, prior_avggaussian, prior_gaussiangaussian, prior_checkerboardnoise
from trainers.utils import rollout_prb

from modelComp.utils import ACT_MAPPER, SKIPBLOCK_MAPPER

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.set_float32_matmul_precision('medium')

# following is a gpu mig bug fix
if "MIG" in subprocess.check_output(["nvidia-smi", "-L"], text=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('MIG GPU detected, using GPU 0')
else:
    print('No MIG GPU detected, using all available GPUs')


class DotDict(dict):
    def __init__(self, mapping=None):
        super().__init__()
        mapping = mapping or {} 
        for key, value in mapping.items():
            self[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not in config")

    def __setattr__(self, key, value):
        self[key] = value

def load_yaml_as_dotdict(filepath):
    with open(filepath, "r") as file:
        data = yaml.safe_load(file) or {}  # for if yaml empty lined
    return DotDict(data)

def read_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CB", type=str, default="std")
    parser.add_argument("--CD", type=str, default="std")
    parser.add_argument("--CM", type=str, default="std")
    parser.add_argument("--CT", type=str, default="std")
    parser.add_argument("--trainer", type=str, default="MTT")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--calc", type=str, required=True)
    parser.add_argument("--fm_samples", type=int)
    parser.add_argument("--dsplit", type=int, nargs='+', required=True) 
    args = parser.parse_args()

    if os.path.exists("conf/base/" + args.CB + ".yaml"):
        cb = load_yaml_as_dotdict("conf/base/" + args.CB + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CB}.yaml not found.")
    if os.path.exists("conf/data/" + args.CD + ".yaml"):
        cd = load_yaml_as_dotdict("conf/data/" + args.CD + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CD}.yaml not found.")
    if os.path.exists("conf/model/" + args.CM + ".yaml"):
        cm = load_yaml_as_dotdict("conf/model/" + args.CM + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CM}.yaml not found.")
    if os.path.exists("conf/training/" + args.CT + ".yaml"):
        ct = load_yaml_as_dotdict("conf/training/" + args.CT + ".yaml")
    else:
        raise FileNotFoundError(f"Config file {args.CT}.yaml not found.")
    
    if args.out != None:
        cb.folder_out = args.out.replace("/", "") + "/"
        #print('args flag')
    if os.path.exists(cb.save_path + cb.folder_out) == False:
        os.makedirs(cb.save_path + cb.folder_out, exist_ok=True)
    if args.out != None:
        cb.wandb_name = args.out
    if args.calc not in ["ss", "ssms", "ms", "spectra", "msspectra", "all"]:
        raise ValueError("Invalid calculation type specified. Choose from 'ss', 'ssms', 'ms', 'spectra', 'msspectra' or 'all'.")
    if args.fm_samples is None and args.trainer == "FM":
        raise ValueError("For FM trainer, --fm_samples argument must be provided.")
    if args.dsplit == 0:
        raise ValueError("dsplit must be greater than 0.")
    #if not isinstance(args.dsplit, int) and not isinstance(args.dsplit, list):
    #    raise ValueError("dsplit must be an integer or a list of integers.")
    return cb, cd, cm, ct, args.trainer, args.model_path, args.calc, args.fm_samples, args.dsplit


class ModelValidationPlot:
    def __init__(self, cb, cd, cm, ct, trainer, model_path, calc, fm_samples=1, dsplit=0):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.trainer = trainer
        self.model_path = model_path
        self.calc = calc
        self.samples = fm_samples
        self.dsplit = dsplit if isinstance(dsplit, list) else [dsplit]
        
        #self.ct.int_steps = 20  # for FM inference

        self.init_modules()

    def init_modules(self):
        # load model
        self.load_model()
        # load dataloaders
        self.load_dataloaders()

    def load_model(self):
        if self.cm.model_name == "FluidGPT":
            if self.trainer == "MTT":
                from modelComp.FluidGPT_B import FluidGPT_B
                self.model = FluidGPT_B(emb_dim=self.cm.emb_dim,
                            data_dim=[self.ct.batch_size, self.cm.temporal_bundling, self.cm.in_channels, self.cd.resample_shape, self.cd.resample_shape],
                            patch_size=(self.cm.patch_size, self.cm.patch_size),
                            hiddenout_dim=self.cm.hiddenout_dim,
                            depth=self.cm.depth,
                            stage_depths=self.cm.stage_depths,
                            num_heads=self.cm.num_heads,
                            window_size=self.cm.window_size,
                            use_flex_attn=self.cm.use_flex_attn,
                            act=ACT_MAPPER[self.cm.act],
                            skip_connect=SKIPBLOCK_MAPPER[self.cm.skipblock],
                            gradient_flowthrough=self.cm.gradient_flowthrough,
                            ).cuda()  
            elif self.trainer == "FM":
                if self.cm.depth == 3:
                    from modelComp.FluidGPT_FM_d3 import FluidGPT_FM
                    print("---\nUSING DEPTH3 MODEL\n----")
                elif self.cm.depth == 2:
                    from modelComp.FluidGPT_FM import FluidGPT_FM
                else:
                    raise ValueError('MODEL DEPTH NOT RECOGNIZED')
                self.model = FluidGPT_FM(emb_dim=self.cm.emb_dim,
                                data_dim=[self.ct.batch_size, self.cm.temporal_bundling, self.cm.in_channels, self.cd.resample_shape, self.cd.resample_shape],
                                embedder_type=self.cm.embedder_type,
                                patch_size=(self.cm.patch_size, self.cm.patch_size),
                                hiddenout_dim=self.cm.hiddenout_dim, 
                                flowmatching_emb_dim=self.cm.flowmatching_emb_dim,
                                depth=self.cm.depth,
                                stage_depths=self.cm.stage_depths,
                                num_heads=self.cm.num_heads,
                                window_size=self.cm.window_size,
                                use_flex_attn=self.cm.use_flex_attn,
                                causal_attn=self.cm.causal_attn,
                                act=ACT_MAPPER[self.cm.act],
                                skip_enable=self.cm.skip_enable,
                                skip_connect=SKIPBLOCK_MAPPER[self.cm.skipblock],
                                gradient_flowthrough=self.cm.gradient_flowthrough,
                                enable_final_layer=self.cm.final_layer
                                ).cuda()
            else:
                raise ValueError("Trainer not recognized in model loading.")
        else:
            raise ValueError('MODEL NOT RECOGNIZED')   
        #self.model.load_state_dict(torch.load_state_dict(self.model_path))
        #print keys in the checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        #print("Checkpoint keys:", checkpoint.keys())
        #for i in range(len(checkpoint['state_dict'].keys())):
            #print(checkpoint['state_dict'].keys()[i], self.model.state_dict().keys()[i])
        new_model_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            
            new_key = key.replace('model.', '', 1)
            new_model_state_dict[new_key] = value
        self.model.load_state_dict(new_model_state_dict)
        self.model.eval()
        print("Model loaded from", self.model_path)

        #raise NotImplementedError("Not yet implemented.")

    def load_dataloaders(self):
        
        # assume preprocessing has already been done !
        
        self.val_datasets = []
        self.valtraj_datasets = []
        self.val_samplers = []
        self.val_loaders = []
        self.valtraj_loaders = []

        means, stds, sizes = [], [], []
                
        for item in self.cd.datasets:
            preproc_savepath = str(self.cb.data_base + 'preproc_' + item["name"])
            if self.trainer == "MTT":
                dataset_SS = DiskDatasetDiv(preproc_savepath, temporal_bundling=self.cm.temporal_bundling, forward_steps=1, fulltrajmode=False)
            #dataset_FS = DiskDatasetDiv(preproc_savepath, temporal_bundling=self.cm.temporal_bundling, forward_steps=self.ct.forward_steps_loss)
            elif self.trainer == "FM":
                dataset_SS = DiskDatasetDivFM(preproc_savepath, temporal_bundling=self.cm.temporal_bundling,
                noisetype=self.ct.noise_type, from_frame=self.ct.from_frame, fulltrajmode=False, sigma_time=self.ct.sigma_time if self.ct.noise_type == 'gaussiangaussian' else None,
                sigma_space=self.ct.sigma_space if self.ct.noise_type == 'gaussiangaussian' else None
                )
            else:
                raise ValueError("Trainer not recognized in validation dataloader.")
            # generate random seed
            seed = self.cd.seed
            #random_seed = random.randint(0, 10000)
            #train_sampler = ZeroShotSamplerReduced(dataset_SS, train_ratio=self.ct.train_ratio, 
            #                                       split="train", seed=seed, skip_timesteps=item["timesample"])
            val_sampler = ZeroShotSamplerReduced(dataset_SS, train_ratio=self.ct.train_ratio, 
                                                 split="val", seed=seed, skip_timesteps=item["timesample"])
            #val_forward_sampler = ZeroShotSamplerReduced(dataset_FS, train_ratio=self.ct.train_ratio, split="val", seed=seed, forward_steps=self.ct.forward_steps_loss)

            #self.train_datasets.append(Subset(dataset_SS, train_sampler.indices))
            self.val_datasets.append(Subset(dataset_SS, val_sampler.indices))
            
            self.valtraj_datasets.append(Subset(dataset_SS, val_sampler.val_trajs))
            
            #self.val_forward_datasets.append(Subset(dataset_FS, val_forward_sampler.indices))
            #self.train_samplers.append(train_sampler)
            self.val_samplers.append(val_sampler)
            #self.val_forward_samplers.append(val_forward_sampler)
            
            #torch.synchronize() 
            split = {
                "name": item["name"],
                "seed": seed,
                #"train_trajs": train_sampler.train_trajs,
                "val_trajs": val_sampler.val_trajs,
                #"val_forward_trajs": val_forward_sampler.val_trajs,
                #"train_idxs": train_sampler.indices,
                "val_idxs": val_sampler.indices,
                #"val_forward_idxs": val_forward_sampler.indices,
            }
            if self.cb.save_on:
                save_split_path = self.cb.save_path + "validation/" + self.trainer + "/" + self.cb.folder_out + "traj_splits/"
                os.makedirs(save_split_path, exist_ok=True)
                save_split_path += "traj_split_" + item["name"] + ".json"
                if save_split_path is None:
                    raise ValueError("ModelCheckpoint callback not found, unable to save trajectory split.")
                with open(save_split_path, "w") as f:
                    json.dump(split, f, indent=0)
                if wandb.run is not None:
                    wandb.save(save_split_path)
            
            mean_i = dataset_SS.avg
            std_i = dataset_SS.std
            N_i = np.prod(dataset_SS.datashape)

            means.append(mean_i)
            stds.append(std_i)
            sizes.append(N_i)
            print("dataset", item["name"], "loaded,", np.prod(dataset_SS.datashape), "elements")

        means = np.array(means)
        stds = np.array(stds)
        sizes = np.array(sizes)

        self.global_mean = np.sum(sizes * means) / np.sum(sizes)
        self.global_std = np.sqrt(np.sum(sizes * (stds**2 + (means - self.global_mean)**2)) / np.sum(sizes))
        if self.ct.normalize:
            for dataset_list in [self.val_datasets, self.valtraj_datasets]: #, self.val_forward_datasets]:
                for subset in dataset_list:
                    subset.dataset.avgnorm = self.global_mean
                    subset.dataset.stdnorm = self.global_std
                
        #self.train_dataset = ConcatDataset(self.train_datasets)
        #self.val_dataset = ConcatDataset(self.val_datasets)
        #self.val_forward_dataset = ConcatDataset(self.val_forward_datasets)
        print("datasets ready.")
               
    def get_dataloader(self, dataset_idx, mode='ss'):
        if mode == 'ss':
            self.batch_size = 512 if "B200" in torch.cuda.get_device_name() else 64
            dataset = self.val_datasets[dataset_idx]
            dataset.dataset.fulltrajmode = False
        elif mode == 'ms' or mode == 'spectra':
            if "B200" in torch.cuda.get_device_name() and not (set([dataset_idx]) & set([0,2])): 
                self.batch_size = 512
            elif "B200" in torch.cuda.get_device_name() and set([dataset_idx]) & set([0,2]): # amira and pdebench incomp
                self.batch_size = 128
            else:
                self.batch_size = 64
            dataset = self.valtraj_datasets[dataset_idx]
            dataset.dataset.fulltrajmode = True
        else:
            raise ValueError("Mode not recognized in dataloader.")
        
        self.batch_size = 1
        print("Using device:", torch.cuda.get_device_name(), "with batch size", self.batch_size)
        return DataLoader(dataset,
                batch_size=self.batch_size, #1, #int(self.ct.batch_size / 8), ################################################################### temporary
                shuffle=True, ###################################################################################3 also temporary
                drop_last=False,
                pin_memory=self.ct.pin_memory, 
                num_workers=self.ct.num_workers, 
                persistent_workers= self.ct.persistent_workers if self.ct.num_workers > 0 else False,
                prefetch_factor=self.ct.prefetch_factor if self.ct.num_workers > 0 else None
            )
    
    def _generate_prior(self, target):
        if self.ct.noise_type == 'puregaussian':
            prior = prior_purenoise(target.clone(), fromframe=self.ct.from_frame)
        elif self.ct.noise_type == 'checkerboard':
            prior = prior_checkerboardnoise(target.clone(), fromframe=self.ct.from_frame)
        elif self.ct.noise_type == 'avggaussian':
            prior = prior_avggaussian(target.clone(), fromframe=self.ct.from_frame, smooth_passes=3)
        elif self.ct.noise_type == 'gaussiangaussian':
            prior = prior_gaussiangaussian(target.clone(), fromframe=self.ct.from_frame, 
                                           sigma_time=self.ct.sigma_time, sigma_space=self.ct.sigma_space)
        else:
            raise ValueError(f"Unknown noisetype {self.ct.noise_type}")
        return prior

    def rollout_tensor(self, traj = None):
        # Get indices for each dataset
        if traj is None:
            raise ValueError("Trajectory index must be provided for rollout error calculation.")
        self.model.eval()
        print("\nStarting Multi-step rollout error calculation...")

        for d, dataset in enumerate(self.valtraj_datasets):
            if d + 1 not in self.dsplit:
                continue
            print(f"\nDataset: {self.cd.datasets[d]['name']}") 
            trajs = self.val_samplers[d].val_trajs
            testtraj = dataset.dataset.get_single_traj(trajs[0])
            testtraj = testtraj.cuda()
            if self.trainer == "MTT":
                testtraj = testtraj.unsqueeze(0) 
            elif self.trainer == "FM":
                testtraj = testtraj.permute(0,2,1,3,4)
            timesteps = testtraj.shape[1]
            print("Timesteps in trajectory:", timesteps)
            #dataloader = self.get_dataloader(d, mode='ms')
            self.samples = 1
            #print("Timesteps:", timesteps)
            individual_rrmse_errors = [list() for _ in range(timesteps)]
            individual_rae_errors = [list() for _ in range(timesteps)]
            #for i, batch in enumerate(dataloader):
            batch = dataset.dataset.get_single_traj(trajs[traj])
            if self.trainer == "MTT":
                batch = batch.unsqueeze(0) 
            #elif self.trainer == "FM":
            #    batch = batch.permute(0,2,1,3,4)
            yfull = batch.cuda()
            with torch.no_grad():
                y = yfull.clone()
                if self.trainer == "MTT":
                    y = y.cuda()
                    #print("y shape:", y.shape)
                    x = y[:,:self.cm.temporal_bundling]
                    yhat_rollout = rollout_det(x, self.model, y.shape[1] // self.cm.temporal_bundling + 1)
                    #print(yhat_rollout.shape, y.shape)
                elif self.trainer == "FM":
                    y = y.cuda()
                    y = y.squeeze(1)
                    #print(y.shape)

                    #print("y shape:", y.shape)
                    x = self._generate_prior(y[:,:,:self.cm.temporal_bundling])
                    #print(x.shape)
                    yhat_rollout = rollout_prb(x, self.model, int(np.ceil((y.shape[2] - self.ct.from_frame) / (self.cm.temporal_bundling - self.ct.from_frame))), 
                                self.ct.int_steps, self.ct.from_frame, noisetype=self.ct.noise_type,
                                sigma_time = self.ct.sigma_time if self.ct.noise_type == 'gaussiangaussian' else None, sigma_space = self.ct.sigma_space if self.ct.noise_type == 'gaussiangaussian' else None)
                    #print(yhat_rollout.shape)
                    #print("Rollout shape before permute:", yhat_rollout.shape)
                    yhat_rollout, y = yhat_rollout.permute(0,2,1,3,4), y.permute(0,2,1,3,4)
                else:
                    raise ValueError("Trainer not recognized in rollout error calculation.")
                yhat_rollout = yhat_rollout[:, :y.shape[1]] 
                
                #y, yhat_rollout = y.squeeze(0), yhat_rollout.squeeze(0)
                unnorm_yhat = yhat_rollout * self.global_std + self.global_mean
                unnorm_y = y * self.global_std + self.global_mean
                
                #print("Rollout shape:", unnorm_yhat.shape)
                #print("Ground truth shape:", unnorm_y.shape)
                diff = unnorm_yhat - unnorm_y
                #print(diff.shape)
                #raise NotImplementedError("Temporary stop for debugging.")
                # Calculate the error for each timestep in the rollout
                #print(yhat_rollout.shape)
                reduce_dims = tuple(range(2, diff.ndim)) # reduce over all but batch and time
                se_sum = diff.pow(2).sum(dim=reduce_dims)         
                ae_sum = diff.abs().sum(dim=reduce_dims)           
                y2_sum = unnorm_y.pow(2).sum(dim=reduce_dims)      
                yabs_sum = unnorm_y.abs().sum(dim=reduce_dims)
                batch_rrmse = torch.sqrt(se_sum / y2_sum)
                batch_rae = ae_sum / yabs_sum
            
                for b in range(batch_rrmse.shape[0]): 
                    for t in range(batch_rrmse.shape[1]): 
                        individual_rrmse_errors[t].append(batch_rrmse[b, t].item())
                        individual_rae_errors[t].append(batch_rae[b, t].item())
            """
            #print(len(individual_rrmse_errors), len(individual_rrmse_errors[0]))
            save_error_path = self.cb.save_path + "validation/" + self.trainer + "/" + self.cb.folder_out + "ms_error/"
            os.makedirs(save_error_path, exist_ok=True)

            individual_rrmse_file_path = save_error_path + "individual_rollout_rrmse_" + self.cd.datasets[d]["name"] + ".json"
            with open(individual_rrmse_file_path, "w") as f:
                json.dump(individual_rrmse_errors, f, indent=2)
            
            individual_rae_file_path = save_error_path + "individual_rollout_rae_" + self.cd.datasets[d]["name"] + ".json"
            with open(individual_rae_file_path, "w") as f:
                json.dump(individual_rae_errors, f, indent=2)
            """

            mean_rrmse_per_timestep = [np.mean(errors) for errors in individual_rrmse_errors]
            mean_rae_per_timestep = [np.mean(errors) for errors in individual_rae_errors]
            
            """
            mean_rrmse_file_path = save_error_path + "ms_error_" + self.cd.datasets[d]["name"] + ".txt"
            with open(mean_rrmse_file_path, "w") as f:
                for t, error in enumerate(mean_rrmse_per_timestep):
                    f.write(f"Timestep {t}: Avg Relative RMSE: {error}\n")
                for t, error in enumerate(mean_rae_per_timestep):
                    f.write(f"Timestep {t}: Avg Relative AE: {error}\n")
            print()
            """
            print(f"MS error {self.cd.datasets[d]['name']} calculation done.")


        del individual_rrmse_errors
        del individual_rae_errors
        return unnorm_yhat, unnorm_y, trajs[traj], mean_rae_per_timestep, mean_rrmse_per_timestep


if __name__ == "__main__":
    traj_list =  [0,1,2,3,4,5,6,7,8,9]
    dataset_list = [1,2,3,4,5,6,7,8,9]
    models_cfg = [
    dict(
        label    = "AR-d2",
        trainer  = "MTT",
        CB = "surf-high", CD = "spike-preprocAll",
        CM = "ar-final-d2", CT = "ar-final",
        ckpt = "models/ar-d2-9/epoch=0048-val_SS_loss_checkpoint=0.0027003738.ckpt",
        #dsplit = [dataset_idx],
    ),
    dict(
        label    = "AR-d3",
        trainer  = "MTT",
        CB = "surf-high", CD = "spike-preprocAll",
        CM = "ar-final-d3", CT = "ar-final",
        ckpt = "models/ar-d3-6/epoch=0055-val_SS_loss_checkpoint=0.0028912849.ckpt",
        #dsplit = [dataset_idx],
    ),
    dict(
        label    = "FM-d2",
        trainer  = "FM",
        CB = "surf-high", CD = "spike-preprocAll",
        CM = "fm-final-d2", CT = "fm-final",
        ckpt = "models/fm-d2-9/epoch=0098-val_SS_loss_checkpoint=0.0002288722.ckpt",
        #dsplit = [dataset_idx],
        fm_samples = 1,
    ),
    dict(
        label    = "FM-d3",
        trainer  = "FM",
        CB = "surf-high", CD = "spike-preprocAll",
        CM = "fm-final-d3", CT = "fm-final",
        ckpt = "models/fm-d3-6/epoch=0098-val_SS_loss_checkpoint=0.0002519532.ckpt",
        #dsplit = [dataset_idx],
        fm_samples = 1,
    )
    ]
    for dsplit_idx in dataset_list:
        for traj_idx in traj_list:
            results = []
            for cfg in models_cfg:
                print(f"\n{'='*50}\nLoading: {cfg['label']}\n{'='*50}")
                cb = load_yaml_as_dotdict(f"conf/base/{cfg['CB']}.yaml")
                cd = load_yaml_as_dotdict(f"conf/data/{cfg['CD']}.yaml")
                cm = load_yaml_as_dotdict(f"conf/model/{cfg['CM']}.yaml")
                ct = load_yaml_as_dotdict(f"conf/training/{cfg['CT']}.yaml")

                mv = ModelValidationPlot(
                    cb, cd, cm, ct,
                    trainer    = cfg['trainer'],
                    model_path = cfg['ckpt'],
                    calc       = 'ms',
                    fm_samples = cfg.get('fm_samples', 1),
                    dsplit     = dsplit_idx,
                )

                traj_true_unnorm, traj_pred_unnorm, actual_traj_idx, rae_errors, rrmse_errors = mv.rollout_tensor(traj_idx)
                results.append((cfg['label'], actual_traj_idx, traj_pred_unnorm.cpu(), traj_true_unnorm.cpu(), rae_errors, rrmse_errors))
                del mv
                torch.cuda.empty_cache()

            print("\nAll models done, now plotting...")

            T = traj_true_unnorm.shape[1]
            timesteps = [0, 1, 2, (T) // 3, 2 * (T) // 3, T - 1]

            trajtrue_denorm = results[0][2].squeeze()

            fig, axes = plt.subplots(1, len(timesteps), figsize=(2 * len(timesteps), 3))
            ax = axes.flatten()
            for col, t in enumerate(timesteps):
                frame = trajtrue_denorm[t]  # (C, H, W)

                vx = frame[0]
                vy = frame[1]
                data = np.sqrt(vx**2 + vy**2)
                im = ax[col].imshow(data.numpy(), cmap="viridis", origin="lower")
                ax[col].set_xticks([])
                ax[col].set_yticks([])

                ax[col].set_title(f"t={t}", fontsize=11)
                if col == 0:
                    ax[col].set_ylabel(r"$\sqrt{x^2+y^2}$", fontsize=12, rotation=90, labelpad=20)

            fig.suptitle(f"Ground truth radial velocity components\nfor trajectory {results[0][1]} from dataset {cd.datasets[dsplit_idx - 1]['name']}", fontsize=24)
            #fig.supylabel(rf"$\sqrt{{x^2+y^2}}$", rotation=90)
            plt.tight_layout(w_pad=0.1, h_pad=0.1)
            plt.savefig(f'scripts/temp/target_{cd.datasets[dsplit_idx - 1]['name']}_{results[0][1]}.png', dpi=600)
            del fig, axes, ax

            timesteps = [5, 6, 7, (T - 5) // 3 + 5, 2 * (T - 5) // 3 + 5, T - 1]

            fig, axes = plt.subplots(5, len(timesteps), figsize=(2 * len(timesteps), 12))
            #fig.suptitle(f"Radial velocity components for trajectory {traj_idx} from dataset {DATASET_NAME}", fontsize=16)
            row = 0
              # (T, C, H, W)
            for col, t in enumerate(timesteps):
                frame = trajtrue_denorm[t]  # (C, H, W)

                vx = frame[0]
                vy = frame[1]
                data = np.sqrt(vx**2 + vy**2)
                im = axes[row, col].imshow(data.numpy(), cmap="viridis", origin="lower")
                axes[row,col].set_xticks([])
                axes[row, col].set_yticks([])

                axes[row, col].set_title(f"t={t}", fontsize=12)
                if col == 0:
                    axes[row, col].set_ylabel("GT", fontsize=12, rotation=0, labelpad=20)

            
            for row in range(1, 5):
                traj_denorm = results[row-1][3].squeeze()  # (T, C, H, W)
                for col, t in enumerate(timesteps):
                    frame = traj_denorm[t]  # (C, H, W)

                    vx = frame[0]
                    vy = frame[1]
                    data = np.sqrt(vx**2 + vy**2)
                    im = axes[row, col].imshow(data.numpy(), cmap="viridis", origin="lower")
                    axes[row,col].set_xticks([])
                    axes[row, col].set_yticks([])
                    #if row == 4:
                        #axes[row, col].set_xlabel(f"Timestep {t}", fontsize=12)
                        #axes[row, col].set_title(f"t={t}", fontsize=11)
                    if col == 0:
                        axes[row, col].set_ylabel(f"{results[row-1][0]}", fontsize=12, rotation=0, labelpad=20)
                    error = results[row-1][5][t]
                    axes[row, col].text(
                        0.98, 0.98,
                        f"Error: {error:.4f}",
                        transform=axes[row, col].transAxes,
                        ha="right",
                        va="top",
                        fontsize=12,
                        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none")
                    )

            fig.suptitle(f"Radial velocity components of model predictions\nfor trajectory {results[0][1]} from dataset {cd.datasets[dsplit_idx - 1]['name']}", fontsize=24)
            fig.supylabel(rf"$\sqrt{{x^2+y^2}}$", rotation=90)
            plt.tight_layout(w_pad=0.15, h_pad=0.15)
            plt.savefig(f'scripts/temp/preds_{cd.datasets[dsplit_idx - 1]['name']}_{results[0][1]}.png', dpi=600)

            # ------------------------------------------------------------------
            # Vorticity plots
            # ------------------------------------------------------------------
            def compute_vorticity(frame):
                # frame: (C, H, W) with channels [vx, vy, ...]
                vx = frame[0].numpy()
                vy = frame[1].numpy()
                dvy_dx = np.gradient(vy, axis=1)   # d(vy)/dx  -> along W
                dvx_dy = np.gradient(vx, axis=0)   # d(vx)/dy  -> along H
                return dvy_dx - dvx_dy

            # --- single-row ground-truth vorticity panel ---
            timesteps = [0, 1, 2, (T) // 3, 2 * (T) // 3, T - 1]

            fig, axes = plt.subplots(1, len(timesteps), figsize=(2 * len(timesteps), 3))
            ax = axes.flatten()
            for col, t in enumerate(timesteps):
                frame = trajtrue_denorm[t]  # (C, H, W)
                vort = compute_vorticity(frame)
                vmax = np.abs(vort).max()
                im = ax[col].imshow(vort, cmap="viridis", origin="lower")
                ax[col].set_xticks([])
                ax[col].set_yticks([])

                ax[col].set_title(f"t={t}", fontsize=11)
                if col == 0:
                    ax[col].set_ylabel(r"$\omega$", fontsize=12, rotation=90, labelpad=20)

            fig.suptitle(f"Ground truth vorticity\nfor trajectory {results[0][1]} from dataset {cd.datasets[dsplit_idx - 1]['name']}", fontsize=24)
            #fig.supylabel(r"$\omega = \partial_x v_y - \partial_y v_x$", rotation=90)
            plt.tight_layout(w_pad=0.1, h_pad=0.1)
            plt.savefig(f'scripts/temp/vort_target_{cd.datasets[dsplit_idx - 1]['name']}_{results[0][1]}.png', dpi=600)
            del fig, axes, ax

            # --- 5-row ground-truth + model-prediction vorticity panel ---
            timesteps = [5, 6, 7, (T - 5) // 3 + 5, 2 * (T - 5) // 3 + 5, T - 1]

            fig, axes = plt.subplots(5, len(timesteps), figsize=(2 * len(timesteps), 12))
            row = 0
            for col, t in enumerate(timesteps):
                frame = trajtrue_denorm[t]  # (C, H, W)
                vort = compute_vorticity(frame)
                vmax = np.abs(vort).max()
                im = axes[row, col].imshow(vort, cmap="viridis", origin="lower")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

                axes[row, col].set_title(f"t={t}", fontsize=12)
                if col == 0:
                    axes[row, col].set_ylabel("GT", fontsize=12, rotation=0, labelpad=20)

            for row in range(1, 5):
                traj_denorm = results[row - 1][3].squeeze()  # (T, C, H, W)
                for col, t in enumerate(timesteps):
                    frame = traj_denorm[t]  # (C, H, W)
                    vort = compute_vorticity(frame)
                    vmax_row = np.abs(vort).max()
                    im = axes[row, col].imshow(vort, cmap="viridis", origin="lower")
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    if col == 0:
                        axes[row, col].set_ylabel(f"{results[row - 1][0]}", fontsize=12, rotation=0, labelpad=20)
                    error = results[row - 1][5][t]
                    axes[row, col].text(
                        0.98, 0.98,
                        f"Error: {error:.4f}",
                        transform=axes[row, col].transAxes,
                        ha="right",
                        va="top",
                        fontsize=12,
                        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none")
                    )

            fig.suptitle(f"Vorticity of model predictions\nfor trajectory {results[0][1]} from dataset {cd.datasets[dsplit_idx - 1]['name']}", fontsize=24)
            fig.supylabel(r"$\omega$", rotation=90)
            plt.tight_layout(w_pad=0.15, h_pad=0.15)
            plt.savefig(f'scripts/temp/vort_preds_{cd.datasets[dsplit_idx - 1]['name']}_{results[0][1]}.png', dpi=600)
            raise NotImplementedError("Temporary stop for debugging.")
        print(f"Done with dataset {dsplit_idx}")
S