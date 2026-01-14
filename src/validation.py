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


for spike:
python3 src/validation.py --trainer MTT --CB spike-high --CD spike-preprocAll --CM ar-semifinal --CT ar-semifinal --out test --model_path /data/fluidgpt/val_models/ar_epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt --calc 
ar: --model_path /data/fluidgpt/val_models/ar_epoch=0048-val_SS_loss_checkpoint=0.004346.ckpt
fm: --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt
python3 src/validation.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-semifinal --CT fm-semifinal --out test --model_path /data/fluidgpt/val_models/fm_epoch=0112-val_SS_loss_checkpoint=0.000363.ckpt --calc 
fm-val-b200-0 

NOTES:
screens when workspace!
calculate only next frame error instead of next timeblock?
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
    if args.calc not in ["ss", "ssms", "ms", "spectra", "all"]:
        raise ValueError("Invalid calculation type specified. Choose from 'ss', 'ssms', 'ms', 'spectra', or 'all'.")
    
    return cb, cd, cm, ct, args.trainer, args.model_path, args.calc


class ModelValidation:
    def __init__(self, cb, cd, cm, ct, trainer, model_path):
        self.cb = cb
        self.cd = cd
        self.cm = cm
        self.ct = ct
        self.trainer = trainer
        self.model_path = model_path

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
                from modelComp.FluidGPT_FM import FluidGPT_FM
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
        self.val_samplers = []
        self.val_loaders = []

        means, stds, sizes = [], [], []
                
        for item in self.cd.datasets:
            preproc_savepath = str(self.cb.data_base + 'preproc_' + item["name"])
            if self.trainer == "MTT":
                dataset_SS = DiskDatasetDiv(preproc_savepath, temporal_bundling=self.cm.temporal_bundling, forward_steps=1)
            #dataset_FS = DiskDatasetDiv(preproc_savepath, temporal_bundling=self.cm.temporal_bundling, forward_steps=self.ct.forward_steps_loss)
            elif self.trainer == "FM":
                dataset_SS = DiskDatasetDivFM(preproc_savepath, temporal_bundling=self.cm.temporal_bundling,
                noisetype=self.ct.noise_type, from_frame=self.ct.from_frame, sigma_time=self.ct.sigma_time if self.ct.noise_type == 'gaussiangaussian' else None,
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
                save_split_path = self.cb.save_path + "validation/" + self.cb.folder_out + "traj_splits/"
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
            for dataset_list in [self.val_datasets]: #, self.val_forward_datasets]:
                for subset in dataset_list:
                    subset.dataset.avgnorm = self.global_mean
                    subset.dataset.stdnorm = self.global_std
                
        #self.train_dataset = ConcatDataset(self.train_datasets)
        #self.val_dataset = ConcatDataset(self.val_datasets)
        #self.val_forward_dataset = ConcatDataset(self.val_forward_datasets)
        print("datasets ready.")
               
    def get_dataloader(self, dataset_idx):
        return DataLoader(self.val_datasets[dataset_idx],
                batch_size=1, #int(self.ct.batch_size / 8), ################################################################### temporary
                shuffle=False, ###################################################################################3 also temporary
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

    def calculate_ss_error_per_dataset(self):
        self.model.eval()

        for d, dataset in enumerate(self.val_datasets):
            dataloader = self.get_dataloader(d)
            print(f"\nDataset: {self.cd.datasets[d]['name']}") 
            cumulative_se_sum = 0.0
            cumulative_ae_sum = 0.0
            cumulative_y2_sum = 0.0
            cumulative_yabs_sum = 0.0
            total_elements = 0
            individual_rrmse_errors = []
            individual_rae_errors = []

            if self.trainer == "FM":
                steps = self.ct.int_steps
            else:
                steps = None

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if self.trainer == "MTT":
                        x, y = batch
                        x, y = x.cuda(), y.cuda()
                        yhat = self.model(x)
                        #print(yhat.shape)
                    elif self.trainer == "FM":
                        y = batch
                        y = y.cuda()
                        #print(y.shape)
                        yhat = self._generate_prior(y)
                        #print(yhat.shape)
                        for _, t in enumerate(torch.linspace(0, 1, steps+1)[:-1], start=1):
                            pred = self.model(yhat.clone(), t.to(y.device).expand(yhat.size(0)))
                            #print(pred.shape)
                            yhat = yhat.clone() + (1 / steps) * pred.clone()
                        #raise NotImplementedError("Temporary stop for debugging.")
                        
                        y, yhat = y.permute(0,2,1,3,4), yhat.permute(0,2,1,3,4)
                        #print(yhat.shape)
                    else:
                        raise ValueError("Trainer not recognized in ss error calculation.")
                    yhat = yhat.squeeze(0)
                    y = y.squeeze(0)
                    unnorm_yhat = yhat * self.global_std + self.global_mean
                    unnorm_y = y * self.global_std + self.global_mean
                    if self.trainer == "MTT":
                        diff = unnorm_yhat - unnorm_y
                    elif self.trainer == "FM":
                        diff = unnorm_yhat[self.ct.from_frame:] - unnorm_y[self.ct.from_frame:]
                        unnorm_y = unnorm_y[self.ct.from_frame:]
                    #print(diff.shape)

                    se_sum = diff.pow(2).sum().item()  
                    ae_sum = diff.abs().sum().item()  
                    y2_sum = (unnorm_y.pow(2)).sum().item()
                    yabs_sum = unnorm_y.abs().sum().item()
                    num_elements = y.numel()

                    cumulative_se_sum += se_sum
                    cumulative_ae_sum += ae_sum
                    cumulative_y2_sum += y2_sum
                    cumulative_yabs_sum += yabs_sum
                    total_elements += num_elements

                    relative_rrmse = (se_sum / y2_sum) ** 0.5
                    individual_rrmse_errors.append(relative_rrmse)
                    relative_rae = ae_sum / yabs_sum
                    individual_rae_errors.append(relative_rae)
                    if i % 10 == 0:
                        print(f"Progress: {i}/{len(dataloader)} batches", end="\r")
                    if i == 40:
                        break
            

            rrmse = (cumulative_se_sum / cumulative_y2_sum) ** 0.5  
            rae = cumulative_ae_sum / cumulative_yabs_sum 

            print("Relative RMSE:", rrmse)
            print("Relative AE:", rae)

            save_error_path = self.cb.save_path + "validation/" + self.cb.folder_out + "ss_error/"
            os.makedirs(save_error_path, exist_ok=True)
            individual_rrmse_file_path = save_error_path + "individual_rrmse_" + self.cd.datasets[d]["name"] + ".json"
            with open(individual_rrmse_file_path, "w") as f:
                json.dump(individual_rrmse_errors, f, indent=2)
            individual_rae_file_path = save_error_path + "individual_rae_" + self.cd.datasets[d]["name"] + ".json"
            with open(individual_rae_file_path, "w") as f:
                json.dump(individual_rae_errors, f, indent=2)
            save_error_path += "ss_error_" + self.cd.datasets[d]["name"] + ".txt"
            with open(save_error_path, "w") as f:
                f.write(f"Avg Relative RMSE: {rrmse}\n")
                f.write(f"Avg Relative AE: {rae}\n")

            print(f"SS error {self.cd.datasets[d]['name']} calculation done.")
        del individual_rrmse_errors
        del individual_rae_errors
        return 1

        
    def calculate_rollout_error_per_dataset(self):
        # Get indices for each dataset
        self.model.eval()

        for d, dataset in enumerate(self.val_datasets):
            print(f"\nDataset: {self.cd.datasets[d]['name']}") 
            trajs = self.val_samplers[d].val_trajs
            #print(trajs[:10])
            ytest = dataset.dataset.get_single_traj(trajs[0])
            # Lists to store errors for each timestep
            individual_rrmse_errors = [list() for _ in range(ytest.shape[0])]
            individual_rae_errors = [list() for _ in range(ytest.shape[0])]

            for i in range(len(trajs)):

                with torch.no_grad():
                    y = dataset.dataset.get_single_traj(trajs[i])
                    if self.trainer == "MTT":
                        y = y.cuda()
                        x = y.unsqueeze(0)[:,:self.cm.temporal_bundling]
                        yhat_rollout = rollout_det(x, self.model, len(y) // self.cm.temporal_bundling + 1)
                        #print(yhat_rollout.shape, y.shape)
                    elif self.trainer == "FM":
                        y = y.cuda()
                        #print(y.shape)
                        x = self._generate_prior(y[:,:,:self.cm.temporal_bundling])
                        #print(x.shape)
                        yhat_rollout = rollout_prb(x, self.model, int(np.ceil((y.shape[2] - self.ct.from_frame) / (self.cm.temporal_bundling - self.ct.from_frame))), 
                                       self.ct.int_steps, self.ct.from_frame, noisetype=self.ct.noise_type,
                                       sigma_time = self.ct.sigma_time if self.ct.noise_type == 'gaussiangaussian' else None, sigma_space = self.ct.sigma_space if self.ct.noise_type == 'gaussiangaussian' else None)
                        #print(yhat_rollout.shape)
                        yhat_rollout, y = yhat_rollout.permute(0,2,1,3,4), y.permute(0,2,1,3,4)
                        #print(yhat_rollout.shape)
                        #print(yhat_rollout.shape, y.shape)
                        #raise NotImplementedError("Temporary stop for debugging.")
                    else:
                        raise ValueError("Trainer not recognized in rollout error calculation.")
                    yhat_rollout = yhat_rollout.squeeze(0)
                    yhat_rollout = yhat_rollout[:y.shape[0]] 
                    y = y.squeeze(0)
                    #y, yhat_rollout = y.squeeze(0), yhat_rollout.squeeze(0)
                    unnorm_yhat = yhat_rollout * self.global_std + self.global_mean
                    unnorm_y = y * self.global_std + self.global_mean
                    
                    #print("Rollout shape:", unnorm_yhat.shape)
                    #print("Ground truth shape:", unnorm_y.shape)
                    diff = unnorm_yhat - unnorm_y
                    #print(diff.shape)
                    #raise NotImplementedError("Temporary stop for debugging.")
                    # Calculate the error for each timestep in the rollout
                    for t in range(yhat_rollout.shape[0]):  
                        diffslice = diff[t]
                        yslice = unnorm_y[t]
                        
                        se = (diffslice.pow(2)).sum().item()
                        ae = diffslice.abs().sum().item()
                        y2 = (yslice.pow(2)).sum().item()
                        yabs = yslice.abs().sum().item()
                        batch_rrmse = (se / y2) ** 0.5  
                        batch_rae = (ae / yabs) 

                        individual_rrmse_errors[t].append(batch_rrmse)
                        individual_rae_errors[t].append(batch_rae)
                        #print(individual_rrmse_errors)
                if i % 5 == 0:
                    print(f"Progress: {i}/{len(trajs)} trajectories", end="\r")
                if i == 15:
                    break
                    

            save_error_path = self.cb.save_path + "validation/" + self.cb.folder_out + "ms_error/"
            os.makedirs(save_error_path, exist_ok=True)

            # Save individual RMSE errors for this dataset
            individual_rrmse_file_path = save_error_path + "individual_rollout_rrmse_" + self.cd.datasets[d]["name"] + ".json"
            with open(individual_rrmse_file_path, "w") as f:
                json.dump(individual_rrmse_errors, f, indent=2)
            # Save individual AE errors for this dataset
            individual_rae_file_path = save_error_path + "individual_rollout_rae_" + self.cd.datasets[d]["name"] + ".json"
            with open(individual_rae_file_path, "w") as f:
                json.dump(individual_rae_errors, f, indent=2)

            # calculate the mean error per timestep
            mean_rrmse_per_timestep = [np.mean(errors) for errors in individual_rrmse_errors]
            mean_rae_per_timestep = [np.mean(errors) for errors in individual_rae_errors]
            # Save mean errors per timestep for this dataset
            mean_rrmse_file_path = save_error_path + "ms_error_" + self.cd.datasets[d]["name"] + ".txt"
            with open(mean_rrmse_file_path, "w") as f:
                for t, error in enumerate(mean_rrmse_per_timestep):
                    f.write(f"Timestep {t}: Avg Relative RMSE: {error}\n")
                for t, error in enumerate(mean_rae_per_timestep):
                    f.write(f"Timestep {t}: Avg Relative AE: {error}\n")
            print()
            print(f"MS error {self.cd.datasets[d]['name']} calculation done.")
        return 1

    def calculate_spectra_plots_per_dataset(self):
        # not sure yet
        pass

if __name__ == "__main__":
    cb, cd, cm, ct, trainer, model_path, calc = read_command()
    model_validation = ModelValidation(cb, cd, cm, ct, trainer, model_path)
    if calc == "ss":
        model_validation.calculate_ss_error_per_dataset()
    elif calc == "ms":
        model_validation.calculate_rollout_error_per_dataset()
    elif calc == "ssms":
        model_validation.calculate_ss_error_per_dataset()
        model_validation.calculate_rollout_error_per_dataset()
    elif calc == "spectra":
        model_validation.calculate_spectra_plots_per_dataset()
    elif calc == "all":
        model_validation.calculate_ss_error_per_dataset()
        model_validation.calculate_rollout_error_per_dataset()
        model_validation.calculate_spectra_plots_per_dataset()
    else:
        raise ValueError("Invalid calculation type specified. Choose from 'ss', 'ssms', 'ms', 'spectra', or 'all'.")
    pass