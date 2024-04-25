import argparse
import time
import os
import imageio
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import yaml
import glob
from PIL import Image


from nerf import (
    CfgNode,
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
    load_pruned_state_dict
)

def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])

def create_gif(folder_path, frame_duration=200):
    # Makes gif from results pics, puts it in parent folder.
    file_paths = sorted([os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.png')])
    images = [Image.open(file_path) for file_path in file_paths]
    parent_folder = os.path.basename(os.path.dirname(folder_path))
    output_filename = f"{parent_folder}.gif"
    image_directory = os.path.dirname(folder_path)
    output_path = os.path.join(image_directory, output_filename)
    images[0].save(output_path, save_all=True, append_images=images[1:], optimize=False, duration=frame_duration, loop=0)
    print(f'GIF saved at {output_path}')

def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)

def find_checkpoint(logdir):
    checkpoint_files = glob.glob(os.path.join(logdir, '*.ckpt'))
    if not checkpoint_files:
        raise ValueError("No checkpoint files found in the log directory.")
    
    # Sort checkpoint files based on the numeric value in the filename
    checkpoint_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # Get the file with the highest number
    latest_checkpoint = checkpoint_files[-1]
    return latest_checkpoint

def find_config(logdir):
    config_files = glob.glob(os.path.join(logdir, 'config.yml'))
    if not config_files:
        raise ValueError("No config YAML file found in the log directory.")
    elif len(config_files) > 1:
        raise ValueError("More than one config YAML file found in the log directory.")
    else:
        return config_files[0]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", type=str, help="Path to directory containing checkpoint and config files. If none, uses most recent trial in logs"
    )
    configargs = parser.parse_args()

    if configargs.logdir is None:
        # Find the most recently modified folder in 'logs'
        log_parent_dir = 'logs'
        log_folders = sorted(glob.glob(os.path.join(log_parent_dir, '*')), key=os.path.getmtime, reverse=True)
        most_recent_logdir = log_folders[0]
        # Within the most recently modified folder, find the most recently modified sub-folder
        subfolders = sorted(glob.glob(os.path.join(most_recent_logdir, '*')), key=os.path.getmtime, reverse=True)
        configargs.logdir = subfolders[0]
    print(f"\nIn logdir {configargs.logdir}")

    checkpoint = find_checkpoint(configargs.logdir)
    config = find_config(configargs.logdir)

    print(f"\nIn logdir {configargs.logdir}, evaluating:\n{os.path.basename(config)} config and \n{os.path.basename(checkpoint)}\n")

    # Read config file.
    cfg = None
    with open(config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

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

    # Initialize a coarse resolution model.
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
             Nbits = cfg.models.fine.n_bits if type(cfg.models.coarse.n_bits) == int else None,
            symmetric = cfg.models.fine.symmetricquant
            )
        model_fine.to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint, map_location=device)

    # Apply the function to load pruned state dicts
    if "model_coarse_state_dict" in checkpoint:
        load_pruned_state_dict(model_coarse, checkpoint["model_coarse_state_dict"])
        
        print("Coarse Checkpoint keys:")
        for key in checkpoint["model_coarse_state_dict"].keys():
            print(key)

    if model_fine and "model_fine_state_dict" in checkpoint:
        try:
            load_pruned_state_dict(model_fine, checkpoint["model_fine_state_dict"])
        except Exception as e:
            print(f"Error loading fine model: {str(e)}")

    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    render_poses = render_poses.float().to(device)

    # Create directory to save images to.
    resultsfolder = os.path.join(configargs.logdir, "results")
    os.makedirs(resultsfolder, exist_ok=True)

    # Evaluation loop
    times_per_image = []
    for i, pose in enumerate(tqdm(render_poses)):
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            pose = pose[:3, :4]
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
        times_per_image.append(time.time() - start)
        
        savefile = os.path.join(resultsfolder, f"{i:04d}.png")
        imageio.imwrite(
            savefile, cast_to_image(rgb[..., :3], cfg.dataset.type.lower())
        )
            
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")
    create_gif(os.path.join(configargs.logdir, "results"))


if __name__ == "__main__":
    main()