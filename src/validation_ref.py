# import libraries
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

"""
for surf:
conda activate grad312
python src/validation.py --trainer MTT --CB surf-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out ar-semifinal-run-test 
python src/validation.py --trainer FM --CB surf-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-semifinal-run-test
ar: --model_path models/epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt
fm: --model_path models/epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt

python src/validation.py --trainer MTT --CB surf-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out ar-semifinal-run-test --model_path models/epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt --calc 
python src/validation.py --trainer FM --CB surf-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-semifinal-run-test --model_path models/epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --calc 

python src/validation.py --trainer FM --CB surf-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-semifinal-run-test3 --model_path models/epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 3 --calc ssms
python src/validation.py --trainer FM --CB surf-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-semifinal-run-test4 --model_path models/epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 3 --calc ssms

spectr:
python src/validation.py --trainer MTT --CB surf-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out ar-spectr-test --model_path models/epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt --calc spectra --dsplit 4 
python src/validation.py --trainer FM --CB surf-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-spectr-test --model_path models/epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --calc spectra --dsplit 4 --fm_samples 3

for spike:
python3 src/validation.py --trainer MTT --CB spike-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out test --model_path /data/fluidgpt/val_models/ar_epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt --calc 
ar: --model_path /data/fluidgpt/val_models/ar_epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt
fm: --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out test --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --calc 
fm-val-b200-0 

spike testing
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out test --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --dsplit 1 --fm_samples 1 --calc ss
python3 src/validation.py --trainer MTT --CB spike-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out test --model_path /data/fluidgpt/val_models/ar_epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt --calc spectra --dsplit 1 2 3 4 5 6 7 8 9
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out test --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --calc spectra --dsplit 1 2 3 4 5 6 7 8 9 --fm_samples 1

python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ss --dsplit 1
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ms --dsplit 1
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ssms --dsplit 2
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ss --dsplit 3
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ms --dsplit 3
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ssms --dsplit 4 5
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ssms --dsplit 6 7 
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200 --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc ssms --dsplit 8 9

python3 src/validation.py --trainer MTT --CB spike-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out ar-val-b200-spec --model_path /data/fluidgpt/val_models/ar_epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt --calc spectra --dsplit 1 2 3 4 5 6 7 8 9
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 1
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 2
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 3
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 4 5
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 6
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 7 
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out fm-val-b200-spec --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --fm_samples 16 --calc spectra --dsplit 8 9
NOTES:
screens when workspace!
calculate only next frame error instead of next timeblock?

python3 src/validation_ref.py --trainer MTT --CB surf-high --CD spike-preprocAll --CM ar-final-d2 --CT ar-final --out EZref --model_path ../models/ar-d2-9/epoch=0048-val_SS_loss_checkpoint=0.0027003738.ckpt --calc spectra --dsplit 1 2 3 4 5 6 7 8 9 --fm_samples 1
"""


from dataloaders import *
from dataloaders import PREPROC_MAPPER
from dataloaders.utils import get_dataset, ZeroShotSamplerReduced, spatial_resample
#from trainers.utils import make_plot, animate_rollout, magnitude_vel, rollout
from trainers.utils import animate_rollout, magnitude_vel, rollout_det, compute_energy_enstrophy_spectra
from trainers.utils import prior_purenoise, prior_avggaussian, prior_gaussiangaussian, prior_checkerboardnoise
from trainers.utils import rollout_prb

from modelComp.utils import ACT_MAPPER, SKIPBLOCK_MAPPER

warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#1F1F1F'
plt.rcParams['savefig.facecolor'] = '#1F1F1F'
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


class ModelValidation:
    def __init__(self, cb, cd, cm, ct, trainer, model_path, calc, fm_samples=1, dsplit=0):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.trainer = trainer
        self.model_path = model_path
        self.calc = calc
        self.samples = fm_samples
        print(dsplit)
        self.dsplit = dsplit if isinstance(dsplit, list) else [dsplit]
        
        #self.ct.int_steps = 20  # for FM inference

        self.init_modules()

    def init_modules(self):
        # load model
        #self.load_model()
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
        print("Using device:", torch.cuda.get_device_name(), "with batch size", self.batch_size)
        return DataLoader(dataset,
                batch_size=self.batch_size, #1, #int(self.ct.batch_size / 8), ################################################################### temporary
                shuffle=False, ###################################################################################3 also temporary
                drop_last=False,
                pin_memory=self.ct.pin_memory, 
                num_workers=self.ct.num_workers, 
                persistent_workers= self.ct.persistent_workers if self.ct.num_workers > 0 else False,
                prefetch_factor=self.ct.prefetch_factor if self.ct.num_workers > 0 else None
            )
    
    def calc_spectra(self, stacked_pred, dataset_name):
        """
        stacked_pred: [B, T, 2, Nx, Ny]
        """
        B, T, _, Nx, Ny = stacked_pred.shape

        t1 = 5
        t2 = 20
        
        u0 = stacked_pred[:, t1, 0]
        v0 = stacked_pred[:, t1, 1]
        
        u1 = stacked_pred[:, t2, 0]
        v1 = stacked_pred[:, t2, 1]
        #print(u0.shape, v0.shape)
        k0, E0, Z0 = self.calc_spectra2(u0, v0, dataset_name)
        k1, E1, Z1 = self.calc_spectra2(u1, v1, dataset_name)
        
        if dataset_name == "pdebench-incomp" or dataset_name == "amira":
            t3 = T - 1
            u2 = stacked_pred[:, t3, 0]
            v2 = stacked_pred[:, t3, 1]
            k2, E2, Z2 = self.calc_spectra2(u2, v2, dataset_name)

        return ((k0, E0, Z0), (k1, E1, Z1), (k2, E2, Z2)) if dataset_name == "pdebench-incomp" or dataset_name == "amira" else ((k0, E0, Z0), (k1, E1, Z1))
    
    def calc_spectra2(self, u, v, dataset_name="", Lx=1.0, Ly=1.0):
        #print(u.shape, v.shape)
        assert u.shape == v.shape
        #print(u.shape, v.shape)
        device = u.device

        B, nx, ny = u.shape
        dx = Lx / nx
        dy = Ly / ny

        if dataset_name == "pdebench-incomp":
            wx = torch.hann_window(nx, periodic=False, device=device)
            wy = torch.hann_window(ny, periodic=False, device=device)
            window = wx[:, None] * wy[None, :]
            u = u * window
            v = v * window

        kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")

        K = torch.sqrt(KX**2 + KY**2)               
        K_flat = K.flatten()                        

        u_hat = torch.fft.fft2(u, dim=(-2, -1))
        v_hat = torch.fft.fft2(v, dim=(-2, -1))
        #print(u_hat.shape, v_hat.shape)
        E_k = 0.5 * (u_hat.abs()**2 + v_hat.abs()**2)

        w_hat = 1j * (KX * v_hat - KY * u_hat)
        Z_k = 0.5 * w_hat.abs()**2
        #print(E_k.shape, Z_k.shape)
        E_flat = E_k.reshape(B, -1)
        Z_flat = Z_k.reshape(B, -1)

        n_bins = nx // 2
        k_max = K_flat.max()
        k_bins = torch.linspace(0, k_max, n_bins + 1, device=device)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        bin_idx = torch.bucketize(K_flat, k_bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid]

        E_flat = E_flat[:, valid]
        Z_flat = Z_flat[:, valid]

        bin_idx = bin_idx.unsqueeze(0).expand(B, -1)

        E_spec = torch.zeros(B, n_bins, device=device)
        Z_spec = torch.zeros(B, n_bins, device=device)
        counts = torch.zeros(n_bins, device=device)

        E_spec.scatter_add_(1, bin_idx, E_flat)
        Z_spec.scatter_add_(1, bin_idx, Z_flat)

        counts.scatter_add_(0, bin_idx[0], torch.ones_like(bin_idx[0], dtype=torch.float))
        counts[counts == 0] = 1.0 

        E_spec /= counts
        Z_spec /= counts

        return k_centers, E_spec, Z_spec

    def calculate_spectra_plots_per_dataset(self):
        #self.model.eval()
        print("\nStarting Physics error calculation...")

        for d, dataset in enumerate(self.valtraj_datasets):
            if d + 1 not in self.dsplit:
                continue
            
            print(f"\nDataset: {self.cd.datasets[d]['name']}") 
            trajs = self.val_samplers[d].val_trajs
            testtraj = dataset.dataset.get_single_traj(trajs[0])
            testtraj = testtraj.cuda()
            if self.trainer == "MTT":
                testtraj = testtraj.unsqueeze(0) 

            if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                (k0, E0, Z0), (k1, E1, Z1), (k2, E2, Z2) = self.calc_spectra(testtraj, dataset_name=self.cd.datasets[d]['name'])
            else:
                (k0, E0, Z0), (k1, E1, Z1) = self.calc_spectra(testtraj, dataset_name=self.cd.datasets[d]['name'])
            #print()
            #print(k0.shape, E0.shape, Z0.shape)
            #raise NotImplementedError("Temporary stop for debugging.")
            #print(trajs[:10])
            timesteps = dataset.dataset.ts
            dataloader = self.get_dataloader(d, mode='spectra')
            #print(ytest.shape)
            if self.trainer == "MTT":
                self.samples = 1
                #print("Timesteps:", timesteps)
                E_errors_t0 = [list() for r in range(len(k0))]
                Z_errors_t0 = [list() for r in range(len(k0))]
                E_errors_t1 = [list() for r in range(len(k0))]
                Z_errors_t1 = [list() for r in range(len(k0))]
                if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                    E_errors_t2 = [list() for r in range(len(k0))]
                    Z_errors_t2 = [list() for r in range(len(k0))]
                k_steps = k0.cpu().numpy().tolist()

            for i, batch in enumerate(dataloader):

                torch.cuda.synchronize()
                time_start = time.time()
                yfull = batch.cuda()
                unnorm_y = yfull * self.global_std + self.global_mean

                if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                    (k0ref, E0ref, Z0ref), (k1ref, E1ref, Z1ref), (k2ref, E2ref, Z2ref) = self.calc_spectra(unnorm_y, dataset_name=self.cd.datasets[d]['name'])
                else:
                    (k0ref, E0ref, Z0ref), (k1ref, E1ref, Z1ref) = self.calc_spectra(unnorm_y, dataset_name=self.cd.datasets[d]['name'])
    
                
                diff_E0 =  E0ref
                diff_Z0 =  Z0ref
                diff_E1 =  E1ref
                diff_Z1 =  Z1ref
                if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                    diff_E2 =  E2ref
                    diff_Z2 =  Z2ref

                if self.trainer == "MTT": 
                    for g in range(diff_E0.shape[0]): 
                        for r in range(len(k_steps)):
                            E_errors_t0[r].append(diff_E0[g, r].item())
                            Z_errors_t0[r].append(diff_Z0[g, r].item())
                    for g in range(diff_E1.shape[0]): 
                        for r in range(len(k_steps)):
                            E_errors_t1[r].append(diff_E1[g, r].item())
                            Z_errors_t1[r].append(diff_Z1[g, r].item())
                    if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                        for g in range(diff_E2.shape[0]): 
                            for r in range(len(k_steps)):
                                E_errors_t2[r].append(diff_E2[g, r].item())
                                Z_errors_t2[r].append(diff_Z2[g, r].item())
                torch.cuda.synchronize()
                end_time = time.time()
                print(f"Progress: {i}/{len(dataloader)} batches, samplecount: {self.samples}, Timer: {end_time - time_start:.4f} s", flush=True)

            save_error_path = self.cb.save_path + "validation/" + self.trainer + "/" + self.cb.folder_out + "spectra_error/"
            os.makedirs(save_error_path, exist_ok=True)
            file_path_ksteps = save_error_path + "kref_steps_" + self.cd.datasets[d]["name"] + ".json"
            with open(file_path_ksteps, "w") as f:
                json.dump(k_steps, f, indent=2)
            file_path_Et5 = save_error_path + "Eref_t=5_" + self.cd.datasets[d]["name"] + ".json"
            file_path_Et20 = save_error_path + "Eref_t=20_" + self.cd.datasets[d]["name"] + ".json"
            file_path_Zt5 = save_error_path + "Zref_t=5_" + self.cd.datasets[d]["name"] + ".json"
            file_path_Zt20 = save_error_path + "Zref_t=20_" + self.cd.datasets[d]["name"] + ".json"
            with open(file_path_Et5, "w") as f:
                json.dump(E_errors_t0, f, indent=2)
            with open(file_path_Et20, "w") as f:
                json.dump(E_errors_t1, f, indent=2)
            with open(file_path_Zt5, "w") as f:
                json.dump(Z_errors_t0, f, indent=2)
            with open(file_path_Zt20, "w") as f:
                json.dump(Z_errors_t1, f, indent=2)
            if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                file_path_Etend = save_error_path + "Eref_tfinal_" + self.cd.datasets[d]["name"] + ".json"
                file_path_Ztend = save_error_path + "Zref_tfinal_" + self.cd.datasets[d]["name"] + ".json"
                with open(file_path_Etend, "w") as f:
                    json.dump(E_errors_t2, f, indent=2)
                with open(file_path_Ztend, "w") as f:
                    json.dump(Z_errors_t2, f, indent=2)
            
            if self.trainer == "MTT":
                meanE_per_k_t0 = [np.mean(errors) for errors in E_errors_t0]
                meanZ_per_k_t0 = [np.mean(errors) for errors in Z_errors_t0]
                meanE_per_k_t1 = [np.mean(errors) for errors in E_errors_t1]
                meanZ_per_k_t1 = [np.mean(errors) for errors in Z_errors_t1]
                if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                    meanE_per_k_t2 = [np.mean(errors) for errors in E_errors_t2]
                    meanZ_per_k_t2 = [np.mean(errors) for errors in Z_errors_t2]

            
            meanE_file_path = save_error_path + "mean_Eref_error_" + self.cd.datasets[d]["name"] + ".txt"
            meanZ_file_path = save_error_path + "mean_Zref_error_" + self.cd.datasets[d]["name"] + ".txt"
            with open(meanE_file_path, "w") as f:
                f.write("Timestep 5:\n")
                for r, error in enumerate(meanE_per_k_t0):
                    f.write(f"K {k_steps[r]}: Avg E_spectra Ratio: {error}\n")
                f.write("\nTimestep 20:\n")
                for r, error in enumerate(meanE_per_k_t1):
                    f.write(f"K {k_steps[r]}: Avg E_spectra Ratio: {error}\n")
                if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                    f.write("\nFinal Timestep:\n")
                    for r, error in enumerate(meanE_per_k_t2):
                        f.write(f"K {k_steps[r]}: Avg E_spectra Ratio: {error}\n")
            with open(meanZ_file_path, "w") as f:
                f.write("Timestep 5:\n")
                for r, error in enumerate(meanZ_per_k_t0):
                    f.write(f"K {k_steps[r]}: Avg Z_spectra Ratio: {error}\n")
                f.write("\nTimestep 20:\n")
                for r, error in enumerate(meanZ_per_k_t1):
                    f.write(f"K {k_steps[r]}: Avg Z_spectra Ratio: {error}\n")
                if self.cd.datasets[d]['name'] == "pdebench-incomp" or self.cd.datasets[d]['name'] == "amira":
                    f.write("\nFinal Timestep:\n")
                    for r, error in enumerate(meanZ_per_k_t2):
                        f.write(f"K {k_steps[r]}: Avg Z_spectra Ratio: {error}\n")
            print()
            print(f"Spectra calculation {self.cd.datasets[d]['name']} done.")
        return 1

if __name__ == "__main__":
    cb, cd, cm, ct, trainer, model_path, calc, fm_samples, dsplit = read_command()
    model_validation = ModelValidation(cb, cd, cm, ct, trainer, model_path, calc, fm_samples, dsplit)
    if calc == "ss":
        code = model_validation.calculate_ss_error_per_dataset()
        if code: print("SS Success") 
        else: print("SS Failure")
    elif calc == "ms":
        code = model_validation.calculate_rollout_error_per_dataset()
        if code: print("MS Success") 
        else: print("MS Failure")
    elif calc == "ssms":
        code = model_validation.calculate_ss_error_per_dataset()
        if code: print("SS Success") 
        else: print("SS Failure")
        code = model_validation.calculate_rollout_error_per_dataset()
        if code: print("MS Success") 
        else: print("MS Failure")
    elif calc == "spectra":
        code = model_validation.calculate_spectra_plots_per_dataset()
        if code: print("SP Success") 
        else: print("SP Failure")
    elif calc == "msspectra":
        code = model_validation.calculate_rollout_error_per_dataset()
        if code: print("MS Success") 
        else: print("MS Failure")
        code = model_validation.calculate_spectra_plots_per_dataset()
        if code: print("SP Success") 
        else: print("SP Failure")
    elif calc == "all":
        code = model_validation.calculate_ss_error_per_dataset()
        if code: print("SS Success") 
        else: print("SS Failure")
        code = model_validation.calculate_rollout_error_per_dataset()
        if code: print("MS Success") 
        else: print("MS Failure")
        code = model_validation.calculate_spectra_plots_per_dataset()
        if code: print("SP Success") 
        else: print("SP Failure")
    else:
        raise ValueError("Invalid calculation type specified. Choose from 'ss', 'ssms', 'ms', 'spectra', or 'all'.")
    pass