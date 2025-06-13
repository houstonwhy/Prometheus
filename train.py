"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.

Main file for the project. This will create and run new experiments and load checkpoints from wandb. 
Borrowed part of the code from David Charatan and wandb.
"""
import os
from pathlib import Path
from datetime import datetime
import hydra
# import tyro
import omegaconf
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

import torch
import lightning
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from prometheus.utils import import_str

from prometheus.utils.print_utils import cyan
from prometheus.utils.distributed_utils import is_rank_zero


def run_local(cfg: DictConfig):
    """XX"""
    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        # Dataset & Dataloader
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        # Exp & Training 
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
            OmegaConf.update(cfg, "training", cfg.experiment.training)
            OmegaConf.update(cfg, "validation", cfg.experiment.validation)
            OmegaConf.update(cfg, "losses", cfg.experiment.losses)
        # Network & Algo
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]
            OmegaConf.update(cfg, "network" ,cfg.algorithm.network)

    # Set up the output directory.

    system = import_str(cfg['training']['module'])(cfg)
    # Set up CheckPoint Callback
    checkpoint_path= system.parse_jobname(cfg.training.resume_from_checkpoint)
    checkpoint_callback = system.configure_checkpoint_callback(add_time_step = cfg.save_ckpt_with_timestep)
    if checkpoint_path and cfg.training.resume_weights_only:
        system.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['state_dict'], strict=False)
        print(f'Resume only state_dict from {checkpoint_path}')
   
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:{system.output_dir}"), )
        (system.output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (system.output_dir.parents[1] / "latest-run").symlink_to(system.output_dir, target_is_directory=True)
    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        # resume = cfg.get("resume", None)
        #name = system.job_name_with_t if checkpoint_path is None else None
        name = system.job_name_with_t
        wandb_dir = os.path.join(system.output_dir, 'logs',system.job_name)
        os.makedirs(wandb_dir, exist_ok=True)
        offline = cfg.wandb.mode != "online"
        logger = lightning.pytorch.loggers.WandbLogger(
            name=name,
            save_dir=wandb_dir,
            id=name,
            entity=cfg.wandb.entity, # 
            offline=offline,
            project=system.job_name,
            log_model="all" if not offline else False,
            config=OmegaConf.to_container(cfg),
            tags=cfg.tags
        )
    else:
        logger = None
    
    try:
        num_nodes = int(os.environ["NODE_SIZE"])
    except Exception as e:
        print(f'only 1 node')
        os.environ["NODE_SIZE"] = '1'
        num_nodes = 1

    if cfg.debug:
        lightning.seed_everything(cfg.seed)

    ITER_SAMPLE_NUM = cfg.training.batch_size * len(cfg.training.gpus) * cfg.training.accumulate_grad_batches * num_nodes
    fake_length = ITER_SAMPLE_NUM * (10 if cfg.debug else cfg.training.steps_per_epoch)
    train_dataset = import_str(cfg['dataset']['train']['module'])(**cfg['dataset']['train']['args'],fake_length=fake_length,debug = cfg.debug)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=cfg.training.batch_size,
                    num_workers=cfg.training.num_workers,
                    # num_workers=0,
                    shuffle=True,
                    #timeout=10,
                    pin_memory=False,
                    persistent_workers=False
                    )


    fake_length_val = ITER_SAMPLE_NUM * (10 if cfg.debug else cfg.training.steps_per_epoch)
    val_dataset = import_str(cfg['dataset']['val']['module'])(**cfg['dataset']['val']['args'], fake_length=fake_length_val, debug = cfg.debug)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        num_workers=cfg.validation.num_workers,
        shuffle=cfg.validation.shuffle,
        #timeout=10,
        pin_memory=False,
        persistent_workers=False
        )
    
    # Logger 
    if len(cfg.training.gpus) == 1:
        strategy = "auto"
    elif cfg.training.use_deepspeed:
        strategy = DeepSpeedStrategy(stage = 2)
    else:
        strategy = DDPStrategy(find_unused_parameters=True)
    trainer = Trainer(
            default_root_dir=f"workdir/{cfg.name}/{cfg.experiment._name}",
            logger=logger if logger else False,
            max_steps=cfg.training.max_steps,
            check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
            log_every_n_steps=1,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            precision=cfg.training.precision,
            accelerator='gpu',
            devices=cfg.training.gpus,
            # strategy="deepspeed_stage_2_offload"if len(cfg.training.gpus) > 1 else "auto",
            strategy=strategy,
            benchmark=True,
            gradient_clip_val=cfg.training.gradient_clip_val,
            # track_grad_norm=1,
            detect_anomaly=cfg.debug,
            #detect_anomaly=True,
            num_nodes=num_nodes,
            callbacks=[checkpoint_callback],
            #deterministic=True
        )
    # TODO Add wandb wandb checkpoint callback
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model=system,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path= None if cfg.training.resume_weights_only else checkpoint_path,
                )

@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
        with open_dict(cfg):
            if cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
                cfg.wandb.mode = "offline"

    if "name" not in cfg:
        raise ValueError("must specify a name for the run with command line argument '+name=[name]'")

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)


    run_local(cfg)

if __name__ == "__main__":
    run()
