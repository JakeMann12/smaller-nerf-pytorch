import argparse
import glob
import os
import shutil
import time

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, load_pruned_state_dict)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--prune",
        type=str,
        default = None,
        choices=["coarse", "fine", "both"],
        help="Specify which model to prune: 'coarse', 'fine', or 'both'."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
        print('using cached dataset!')
    else:
        print('NOT USING cached dataset!')
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
            )
            i_train, i_val, i_test = i_split
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    # Extract pruning settings from the configuration
    max_pruning = cfg['pruning']['max_prune_amount']
    prune_increment = cfg['pruning']['prune_increment']
    excluded_layers = cfg['pruning']['excluded_layers']
    times_to_prune = int(max_pruning / prune_increment)
    pruning_intervals = [0.0] + [i / times_to_prune for i in range(1, times_to_prune)]  # Percentages of training completion to apply pruning


    start_iter = 0  # Define start_iter if it's not already defined

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        Nbits = cfg.models.coarse.n_bits if type(cfg.models.coarse.n_bits) == int else None,
        symmetric = cfg.models.coarse.symmetricquant
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            Nbits = cfg.models.fine.n_bits if type(cfg.models.fine.n_bits) == int else None,
            symmetric = cfg.models.fine.symmetricquant
        )
        model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)

    #[os.remove(os.path.join(root, file)) for root, dirs, files in os.walk(logdir) for file in files]
    #print('cleared out dirty files')

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        print(f"Success loading {configargs.load_checkpoint}")
    else:
        if configargs.load_checkpoint != "":
            print(f"{configargs.load_checkpoint} doesn't exist! Wrong path perhaps?")
        else:
            def_check = os.path.join(cfg.dataset.cachedir, "checkpoint00000.ckpt")
            print(f'no checkpoint given. Starting from {def_check}!')
        try:
            checkpoint = torch.load(def_check)###### checkpoint = torch.load("pretrained/")    
        except:
            print(f"YOU HAVEN'T PUT A PRETRAINED 0 it MODEL IN {str(cfg.dataset.cachedir)}")

    # Load state dictionaries with support for both pruned and unpruned models
    if "model_coarse_state_dict" in checkpoint:
        load_pruned_state_dict(model_coarse, checkpoint["model_coarse_state_dict"])
    if model_fine and "model_fine_state_dict" in checkpoint:
        load_pruned_state_dict(model_fine, checkpoint["model_fine_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint.get("iter", 0)
        

    # # TODO: Prepare raybatch tensor if batching random rays
    times_to_prune = 10; pruning_intervals = list(np.linspace(.0,.5,times_to_prune+1))
    print(f"\n{times_to_prune}, \n{pruning_intervals}")
    
    #If we need to preserve pruning in future runs
    weight_mask_c = {}
    weight_mask_f = {}
    if configargs.prune is None:
        for name, module in model_coarse.named_modules():
                if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                    weight_mask_c[name] = torch.tensor(np.where(np.abs(module.weight.data.cpu().detach().numpy()) != 0, 1, 0)).to(device)
        for name, module in model_fine.named_modules():
            if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                weight_mask_f[name] = torch.tensor(np.where(np.abs(module.weight.data.cpu().detach().numpy()) != 0, 1, 0)).to(device)


    for i in trange(start_iter, cfg.experiment.train_iters):
        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        
        #%% TRAINING
        if configargs.prune is not None: 
            # Determine the current fraction of completed training
            total_iters = cfg.experiment.train_iters - start_iter
            current_progress = (i - start_iter) / total_iters

            # Find the appropriate pruning amount based on the current training progress
            current_prune_level = 0
            for threshold in pruning_intervals:
                if current_progress >= threshold:
                    # Update pruning level based on the index in intervals
                    current_prune_level = pruning_intervals.index(threshold) * prune_increment
                else:
                    break

            # Apply structured pruning to each appropriate layer if the current iteration matches a pruning interval
            if current_progress in pruning_intervals:
                # Pruning the coarse model
                if configargs.prune == "coarse" or configargs.prune == "both":
                    for name, module in model_coarse.named_modules():
                        if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                            prune.ln_structured(module, name='weight', amount=current_prune_level, n=1, dim=0)
                            print(f'Pruned coarse {name} to {current_prune_level * 100:.2f}% at {current_progress * 100:.1f}% of training')

                # Pruning the fine model
                if configargs.prune == "fine" or configargs.prune == "both":  # Check if there is a fine model defined
                    for name, module in model_fine.named_modules():
                        if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                            prune.ln_structured(module, name='weight', amount=current_prune_level, n=1, dim=0)
                            print(f'Pruned fine {name} to {current_prune_level * 100:.2f}% at {current_progress * 100:.1f}% of training')
        
        else:
            for name, module in model_coarse.named_modules():
                if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                    module.weight.data = module.weight.data * weight_mask_c[name]
            
            for name, module in model_fine.named_modules():
                if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                    module.weight.data = module.weight.data * weight_mask_f[name]

        #Check the weights anyway
        for name, module in model_coarse.named_modules():
            if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
                non_zero_weights = torch.count_nonzero(module.weight).item()
                writer.add_scalar(f'Non-zero weights/coarse_{name}', non_zero_weights, i)

        # for name, module in model_fine.named_modules():
        #     if isinstance(module, (torch.nn.Linear)) and name not in excluded_layers:
        #         non_zero_weights = torch.count_nonzero(module.weight).item()
        #         writer.add_scalar(f'Non-zero weights/fine_{name}', non_zero_weights, i)


        if USE_CACHED_DATASET:
            #print("USING CACHED DATASET")
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )

            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        else:
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )

        #%% CALCULATE LOSS
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000

        if configargs.prune is not None:
            fine_tune_lr = cfg.optimizer.lr * 0.1
            lr_new = fine_tune_lr * (
                cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
            )
        else:
            lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
            )
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        pruneaddon = "PRUNING" if configargs.prune is not None else ""
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                pruneaddon +
                ": [TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # %% Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, pruneaddon + "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")
            # Log the weights regardless of pruning
            for name, module in model_coarse.named_modules():
                    for param_name, param_value in module.named_parameters():
                        if 'weight' in param_name:  # Ensure only weights are logged
                            writer.add_histogram(f"Coarse {name}/{param_name}", param_value, i)
                            writer.flush()

            for name, module in model_fine.named_modules():
                    for param_name, param_value in module.named_parameters():
                        if 'weight' in param_name:  # Ensure only weights are logged
                            writer.add_histogram(f"Fine {name}/{param_name}", param_value, i)
                            writer.flush()

    print("consolidating files")
    #start iter, cfg.experiment.train_iters
    # {postcoarse}{postfine}
    
    # P- PRUNE, CQ - COARSE QUANT, FQ - FINE QUANT
    exp_name = f"P{configargs.prune}-CQ{cfg.models.coarse.n_bits}-FQ{cfg.models.fine.n_bits}_{str(start_iter/1000)}-{str(cfg.experiment.train_iters/1000)}k"
    prev_ckpt = 'post'+str(configargs.load_checkpoint).split('\\')[2].split('P', 1)[1].split('-', 1)[0] if configargs.load_checkpoint is not None else ""
    new_folder_path = os.path.join(logdir, exp_name + prev_ckpt)
    os.makedirs(new_folder_path, exist_ok=True)

    # Iterate through all items in the original folder
    for item in os.listdir(logdir):
        item_path = os.path.join(logdir, item)
        # Check if the item is a file (not a folder)
        if os.path.isfile(item_path):
            # Move the file to the new folder
            shutil.move(item_path, new_folder_path)
    print(f"Done! Folder {exp_name}")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()
