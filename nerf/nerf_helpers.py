import math
from typing import Optional
import kornia
import torch


def img2mse(img_src, img_tgt):
    return torch.nn.functional.mse_loss(img_src, img_tgt)


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor
):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # UNTESTED, but fairly sure.

    # Shift rays origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def gather_cdf_util(cdf, inds):
    r"""A very contrived way of mimicking a version of the tf.gather()
    call used in the original impl.
    """
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, det=False):
    # TESTED (Carefully, line-to-line).
    # But chances of bugs persist; haven't integration-tested with
    # training routines.

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(-1).unsqueeze(-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, num_samples).to(weights)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

    # Invert CDF
    inds = torch.searchsorted(
        cdf.contiguous(), u.contiguous(), side="right"
    )
    below = torch.max(torch.zeros_like(inds), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)
    orig_inds_shape = inds_g.shape

    cdf_g = gather_cdf_util(cdf, inds_g)
    bins_g = gather_cdf_util(bins, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pdf_2(bins, weights, num_samples, det=False):
    r"""sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [num_samples],
            dtype=weights.dtype,
            device=weights.device,
        )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

# WORKS WITH EVAL
def load_pruned_state_dict(model, state_dict):
    try:
        new_state_dict = {}
        for key, value in state_dict.items():
            if '_orig' in key:
                # Handle pruned weights
                mask_key = key.replace('_orig', '_mask')
                if mask_key in state_dict:
                    pruned_weights = state_dict[key] * state_dict[mask_key]
                    new_key = key.replace('_orig', '')
                    new_state_dict[new_key] = pruned_weights
            elif '_mask' not in key:
                # This handles unpruned weights or any other entries normally
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        model.eval()  # Set model to evaluation mode after loading
    except:
        load_pruned_state_dict_weird(model, state_dict)


def calculate_ssim(tensor1, tensor2): #FOR EVAL
    def tensor_to_image(tensor):
        # Assuming the tensor's size in terms of number of elements matches 3 * 400 * 400
        return tensor.reshape(1, 3, 400, 400)

    img1 = tensor_to_image(tensor1)
    img2 = tensor_to_image(tensor2)

    # Normalize the tensors to [0, 1] if they are not already
    img1 = img1.float() / 255.0
    img2 = img2.float() / 255.0

    # Compute SSIM using the kornia.metrics.ssim function
    window_size = 11
    ssim_map = kornia.metrics.ssim(img1, img2, window_size, max_val=1.0, eps=1e-12, padding='same')

    # Compute mean SSIM over all elements
    mean_ssim = ssim_map.mean().item()

    return mean_ssim


# def load_pruned_state_dict(model, state_dict):
#     new_state_dict = {}
#     keys = []
#     print(f"OG STATE DICT:\n {list(state_dict.keys())}")
#     try:
#         model.load_state_dict(model, state_dict)
#         model.eval()
#     except:
    
#         for key, value in state_dict.items():
#             # Store the original key for debugging
#             keys.append(key)
#             try:
#                 # Handle pruned weights
#                 if '_orig' in key:
#                     mask_key = key.replace('_orig', '_mask')
#                     if mask_key in state_dict:
#                         pruned_weights = state_dict[key] * state_dict[mask_key]
#                         new_key = key.replace('_orig', '')  # Remove '_orig' from the key
#                 else:
#                     pruned_weights = value
#                     new_key = key  # Keep the original key if it's not a pruned weight
#                 # Assign the possibly adjusted weights or the original ones
#                 new_state_dict[new_key] = pruned_weights if '_orig' in key else value
#                 model.load_state_dict(new_state_dict)
#                 model.eval()  # Set model to evaluation mode after loading      


#             except:
#                 print("TRYING EXCEPT CLAUSE")
#                 # Handle pruned weights
#                 if '_orig' in key:
#                     mask_key = key.replace('_orig', '_mask')
#                     if mask_key in state_dict:
#                         pruned_weights = state_dict[key] * state_dict[mask_key]
#                         new_key = key.replace('_orig', '')  # Remove '_orig' from the key
#                 else:
#                     pruned_weights = value
#                     new_key = key  # Keep the original key if it's not a pruned weight
#                 # QUANT: Insert 'linear' before the last part of the key if it doesn't already include 'linear'
#                 if 'linear' not in new_key:
#                     parts = new_key.split('.')
#                     parts.insert(-1, 'linear')
#                     new_key = '.'.join(parts)
#                 # Assign the possibly adjusted weights or the original ones
#                 new_state_dict[new_key] = pruned_weights if '_orig' in key else value
#                 print(f"OLD KEYS: \n{keys}")
#                 print(f"NEW KEYS: \n{list(new_state_dict.keys())}")
#                 model.load_state_dict(new_state_dict)
#                 model.eval()  # Set model to evaluation mode after loading      

    
    


def load_pruned_state_dict_weird(model, state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key  # Start with assuming no change to key

        # Handle pruned weights
        if '_orig' in key:
            mask_key = key.replace('_orig', '_mask')
            if mask_key in state_dict:
                pruned_weights = state_dict[key] * state_dict[mask_key]
                new_key = key.replace('_orig', '')

        # Adjust keys for layers that are expected to contain 'linear'
        # This specifically checks for layer names in the xyz layers
        layer_indicators = ["layers_xyz.", "layer1.", "layers_dir.", "fc_"]
        if any(indicator in new_key for indicator in layer_indicators):
            # Check if 'linear' needs to be added
            if 'linear' not in new_key:
                parts = new_key.split('.')
                # Insert 'linear' before the last part (weight or bias)
                parts.insert(-1, 'linear')
                new_key = '.'.join(parts)

        # Assign the possibly adjusted weights or the original ones
        new_state_dict[new_key] = value if '_orig' not in key else pruned_weights

    # Attempt to load the modified state dictionary into the model
    try:
        model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore non-matching keys
        model.eval()  # Set model to evaluation mode after loading
        print("Model loaded successfully with adjusted keys.")
    except RuntimeError as e:
        print(f"Failed to load model: {e}")








    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
if __name__ == "__main__":

    # # meshgrid_xy
    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
    # print(i)
    # print(j)
    # ii, jj = torch.meshgrid(torch.arange(3), torch.arange(4, 7))
    # print(ii.transpose(-1, -2))
    # print(jj.transpose(-1, -2))
    # ii, jj = meshgrid_xy(torch.arange(3), torch.arange(4, 7))
    # print(ii)
    # print(jj)

    # # dirs (get_rays_np)
    # H, W = 3, 3
    # focal = 10
    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
    # dirs = np.stack([(i - W) * .5 / focal, -(j - H) * .5 / focal, -np.ones_like(i)], -1)
    # print(dirs)
    # ii, jj = meshgrid_xy(torch.arange(3).float(), torch.arange(4, 7).float())
    # dirs_torch = torch.stack([(ii - W) * .5 / focal, -(jj - H) * .5 / focal, -torch.ones_like(ii)], -1)
    # print(dirs_torch)
    # print(np.allclose(dirs, dirs_torch.cpu().numpy()))

    # # rays_o, rays_d (get_rays_np)
    # H, W = 3, 3
    # focal = 10
    # c2w = np.eye(4)
    # c2w[:3, :3] = 2 * c2w[:3, :3]
    # i, j = np.meshgrid(np.arange(3), np.arange(4, 7), indexing='xy')
    # dirs = np.stack([(i - W) * .5 / focal, -(j - H) * .5 / focal, -np.ones_like(i)], -1)
    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    # print(rays_d)
    # print(rays_o)
    # ii, jj = meshgrid_xy(torch.arange(3).float(), torch.arange(4, 7).float())
    # dirs_torch = torch.stack([(ii - W) * .5 / focal, -(jj - H) * .5 / focal, -torch.ones_like(ii)], -1)
    # c2w_torch = torch.eye(4)
    # c2w_torch[:3, :3] = 2 * c2w_torch[:3, :3]
    # rays_d_torch = torch.sum(dirs_torch[..., None, :] * c2w_torch[:3, :3], -1)
    # rays_o_torch = c2w_torch[:3, -1].expand(rays_d_torch.shape)
    # print(rays_d_torch)
    # print(rays_o_torch)
    # print(np.allclose(rays_d, rays_d_torch.cpu().numpy()))
    # print(np.allclose(rays_o, rays_o_torch.cpu().numpy()))

    # # get_rays(_torch) vs get_rays_np
    # H, W = 3, 3
    # focal = 10
    # c2w = np.eye(4)
    # c2w[:3, :3] = 2 * c2w[:3, :3]
    # # c2w_torch = torch.eye(4)
    # # c2w_torch[:3, :3] = 2 * c2w_torch[:3, :3]
    # rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    # c2w_torch = torch.from_numpy(c2w)
    # rays_o_torch, rays_d_torch = get_rays(H, W, focal, c2w_torch)
    # print(np.allclose(rays_o, rays_o_torch.cpu().numpy()))
    # print(np.allclose(rays_d, rays_d_torch.cpu().numpy()))  # Assert fails, values look different.
    # print("Numpy version:")
    # print(rays_d)
    # print("PyTorch version:")
    # print(rays_d_torch.cpu().numpy())

    # Test backprop for sample_pdf()
    bins = torch.rand(2, 4)
    weights = torch.rand(2, 4)
    weights.requires_grad = True
    samples = sample_pdf(bins, weights, 10)
    print(samples)
