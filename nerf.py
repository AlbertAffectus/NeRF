# get my data here:
# https://drive.google.com/drive/folders/1s0_kPLkJ5ueCYbr4-Ddqp5VcRXWdqnbv?usp=sharing

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from ipywidgets import interactive, widgets
from torch.utils.data import DataLoader
import imageio
import argparse
import pickle

RES_PATH = "./results"

# check device availability
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class NeRF(nn.Module):
    """
    Neural Radiance Fields model

    Architecture as described in the paper appendix
    """

    def __init__(
        self, position_embedding_size=10, direction_embedding_size=4, hidden_dim=128
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(position_embedding_size * 6 + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ), nn.Sequential(
                nn.Linear(position_embedding_size * 6 +
                          hidden_dim + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim + 1),
            ), nn.Sequential(
                nn.Linear(direction_embedding_size * 6 +
                          hidden_dim + 3, hidden_dim // 2),
                nn.ReLU(),
            ), nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())
        ])

        self.position_embedding_size = position_embedding_size
        self.direction_embedding_size = direction_embedding_size

        # so we dont have to keep initializing
        self.relu = nn.ReLU()

    def positional_encoding(x, embedding_size):
        """
        Apply positional encoding to input tensor.

        This applies the sinusoidal positional encoding to the input tensor
        as described in the paper.

        Args:
            x (torch.Tensor): The input tensor.
            L (int): The number of encoding functions to use.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        out = [x]
        for i in range(embedding_size):
            for fn in [torch.sin, torch.cos]:
                out.append(fn(2 ** i * x))
        return torch.cat(out, dim=1)

    def forward(self, origins, directions):
        """
        Forward pass of the Neural Radiance Fields model.

        Args:
            origins (torch.Tensor): A tensor of ray origins with shape (batch_size, 3).
            directions (torch.Tensor): A tensor of ray directions with shape (batch_size, 3).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted RGB color (torch.Tensor) 
            and the density (sigma) of the predicted volume (torch.Tensor). Both tensors have shape (batch_size,).
        """
        encoded_pos = self.positional_encoding(
            origins, self.position_embedding_size)
        encoded_dir = self.positional_encoding(
            directions, self.direction_embedding_size)

        h = self.blocks[0](encoded_pos)
        h = self.blocks[1](torch.cat((h, encoded_pos), dim=1))
        c = self.blocks[2](torch.cat((h[:, :-1], encoded_dir), dim=1))
        c = self.block4[3](c)

        sigma = self.relu(h[:, -1])
        return c, sigma

def accumulated_transmittance(alphas):
    """
    Compute the accumulated transmittance along a ray.

    The formula as given by the paper in (3)

    Args:
        alphas (torch.Tensor): The alpha values along the ray, shape (batch_size, num_ray_samples).

    Returns:
        torch.Tensor: The accumulated transmittance along the ray, shape (batch_size, num_ray_samples).
    """
    acc = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((acc.shape[0], 1), device=DEVICE), acc[:, :-1]), dim=-1)


def render_rays(model, origins, directions=None, near_bound=0, far_bound=0.5, num_ray_samples=192, device=DEVICE):
    """
    Render rays using the NeRF model.

    we approximate the integral in (2) by sampling along the ray (approximating the integral with a sum)

    Args:
        model (nn.Module): The NeRF model to use for rendering.
        origins (torch.Tensor): The origin points of the rays, shape (batch_size, 3).
        directions (torch.Tensor): The direction vectors of the rays, shape (batch_size, 3).
        near_bound (float, optional): The near bound of the interval to sample along the rays. Defaults to 0.
        far_bound (float, optional): The far bound of the interval to sample along the rays. Defaults to 0.5.
        num_ray_samples (int, optional): The number of samples to take along each ray. Defaults to 192.
        device (str, optional): The device to use for rendering. Defaults to "cuda".

    Returns:
        torch.Tensor: The rendered color image, shape (batch_size, 3).
    """
    # sample points along rays
    t, delta = sample_points_on_rays(
        origins, directions, near_bound, far_bound, num_ray_samples, device)

    # location of ray at time t
    x, directions_expanded = compute_ray_locations(
        origins, directions, t, num_ray_samples)

    if directions is None:
        colors, sigma = evaluate_model(model, x)
    else:
        colors, sigma = evaluate_model(model, x, directions_expanded)
    weights = calculate_weights(sigma, delta)
    c = apply_weights(colors, weights)

    # add regularization for white background
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1), sigma


def sample_points_on_rays(origins, directions, near_bound, far_bound, num_ray_samples, device):
    # sample t in [near, far] then use these values to compute x along the ray
    t = torch.linspace(near_bound, far_bound, num_ray_samples,
                       device=DEVICE).expand(origins.shape[0], num_ray_samples)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u

    # the width of each sample bin
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10],
                      device=DEVICE).expand(origins.shape[0], 1)), -1)

    return t, delta


def compute_ray_locations(origins, directions, t, num_ray_samples):
    # the location of a ray at time t is the origin + t * direction
    x = origins.unsqueeze(1) + t.unsqueeze(2) * directions.unsqueeze(1)
    directions_expanded = directions.expand(
        num_ray_samples, directions.shape[0], 3).transpose(0, 1)

    return x, directions_expanded


def evaluate_model(model, x, directions=None):
    """
    Evaluate the NeRF model at given points.

    Args:
        model (nn.Module): The NeRF model to use for evaluation.
        x (torch.Tensor): The points to evaluate the model at, shape (batch_size, 3).
        directions (torch.Tensor, optional): The direction vectors of the rays, shape (batch_size, 3). Defaults to None.

    Returns:
        tuple: A tuple containing the predicted colors and sigmas.
    """
    # use the model to predict the color and sigma at each point
    if directions is None:
        zeros = torch.zeros_like(x)
        colors, sigma = model(x.reshape(-1, 3), zeros)
    else:
        colors, sigma = model(x.reshape(-1, 3), directions.reshape(-1, 3))

    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    return colors, sigma


def calculate_weights(sigma, delta):
    # compute the transmittance and weight of each sample
    # the volumetric rendering equation is described in the paper
    alpha = 1 - torch.exp(-sigma * delta)
    weights = accumulated_transmittance(
        1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    return weights


def apply_weights(colors, weights):
    return (weights * colors).sum(dim=1)


@torch.no_grad()
def test(model, near_bound, far_bound, dataset, img_index=0, num_ray_samples=128, h=400, w=400, device=DEVICE):
    chunk_size = 32
    start_index = img_index * h * w
    end_index = (img_index + 1) * h * w

    ray_origins = dataset[start_index:end_index, :3]
    ray_directions = dataset[start_index:end_index, 3:6]
    rgb = dataset[start_index:end_index, 6:9]
    original = rgb.reshape(h, w, 3)

    data = []
    for i in range(int(np.ceil(h / chunk_size))):
        start_chunk = i * w * chunk_size
        end_chunk = (i + 1) * w * chunk_size

        ray_origins_chunk = ray_origins[start_chunk:end_chunk].to(device)
        ray_directions_chunk = ray_directions[start_chunk:end_chunk].to(device)

        predicted_pixels, _ = render_rays(
            model,
            ray_origins_chunk,
            ray_directions_chunk,
            near_bound=near_bound,
            far_bound=far_bound,
            num_ray_samples=num_ray_samples
        )
        data.append(predicted_pixels)

    img = torch.cat(data).data.cpu().numpy().reshape(h, w, 3)
    print(np.max(img))

    # show original and img side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original)
    ax[1].imshow(img)
    plt.show()


def train(model, optimizer, scheduler, data_loader, near_bound=2, far_bound=6, epochs=16,
          num_ray_samples=128, h=400, w=400, device=DEVICE, testing_dataset=None):
    """
    Train the NeRF model on a given dataset.

    Args:
        model (NeRF): A NeRF model to train.
        optimizer (torch.optim.Optimizer): An optimizer to use for the training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): A learning rate scheduler to use.
        data_loader (torch.utils.data.DataLoader): A data loader providing the training dataset.
        near_bound (int, optional): The near bound of the volume to render. Defaults to 2.
        far_bound (int, optional): The far bound of the volume to render. Defaults to 6.
        epochs (int, optional): The number of epochs to train for. Defaults to 16.
        num_ray_samples (int, optional): The number of samples to take along each ray. Defaults to 128.
        h (int, optional): The height of the output image. Defaults to 400.
        w (int, optional): The width of the output image. Defaults to 400.
        device (torch.device, optional): The device on which to run the training. Defaults to DEVICE.

    Returns:
        List[float]: A list of the training losses.
    """
    training_loss = []
    for _ in tqdm(range(epochs)):
        for batch in data_loader:
            origins, directions, actual_pixels = batch[:, :3].to(
                device), batch[:, 3:6].to(device), batch[:, 6:].to(device)
            predicted_pixels, _ = render_rays(
                model, origins, directions, near_bound=near_bound, far_bound=far_bound, num_ray_samples=num_ray_samples)
            loss = ((actual_pixels - predicted_pixels) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        # we want to use a scheduler because NeRF models converge quickly on macro features but slowly on micro features
        scheduler.step()

        if testing_dataset is not None:
            test(model, near_bound, far_bound, testing_dataset,
                 img_index=2, num_ray_samples=num_ray_samples, h=h, w=w)

    return training_loss


def do_training(model, training_dataset, testing_dataset, epochs=12, num_ray_samples=128, h=400, w=400):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 10], gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    losses = train(model, optimizer, scheduler, data_loader, epochs=epochs,
                   num_ray_samples=num_ray_samples, h=h, w=w, testing_dataset=testing_dataset)

    return losses


def load_model(model_path):
    model = NeRF().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    return model


def get_rays(h, w, focal, pose):
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])

    i, j = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2])/K[0][0], -(j - K[1][2]) /
                    K[1][1], -np.ones_like(i)], -1)

    # Convert dirs numpy array to a PyTorch tensor
    dirs = torch.from_numpy(dirs)

    # rotate camera coordinate to world coordinate
    dirs = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
    # broadcast camera origins to have the same shape as dirs
    origins = torch.broadcast_to(pose[:3, -1], (h, w, 3))

    # reshape origins and dirs tensors
    origins = origins.reshape(-1, 3)
    dirs = dirs.reshape(-1, dirs.shape[-1])

    origins = origins.to(DEVICE)
    dirs = dirs.to(DEVICE)

    return origins, dirs


# from https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
def trans_t(t):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def rot_phi(phi):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def rot_theta(th):
    return torch.tensor([
        [torch.cos(th), 0, -torch.sin(th), 0],
        [0, 1, 0, 0],
        [torch.sin(th), 0, torch.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    pose = trans_t(radius)
    pose = rot_phi(torch.tensor(phi / 180. * np.pi)) @ pose
    pose = rot_theta(torch.tensor(theta / 180. * np.pi)) @ pose
    pose = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0],
                        [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ pose
    return pose


def render_video(model, h, w, focal, n_samples=128, batch_size=1024, phi_min=0, phi_max=120, out_path='video.mp4'):
    frames = []
    for th in tqdm(np.linspace(0., 720., 240, endpoint=False)):
        phi_range = phi_max - phi_min
        phi = phi_min + phi_range * 0.5 * (1 + np.sin(th / 180. * np.pi))
        pose = pose_spherical(th, phi, 4.)
        rays_o, rays_d = get_rays(h, w, focal, pose[:3, :4])
        
        # Initialize an empty array for the final image
        img_full = np.zeros((h * w, 3))

        # Process in smaller batches to save memory
        for i in range(0, h * w, batch_size):
            batch_rays_o = rays_o[i:i + batch_size]
            batch_rays_d = rays_d[i:i + batch_size]
            
            # Render the rays for the current batch
            rgb, _ = render_rays(model, batch_rays_o, batch_rays_d, near_bound=2., far_bound=6., num_ray_samples=n_samples)
            img_batch = torch.clamp(rgb, 0, 1).cpu().detach().numpy()
            
            # Update the full image with the rendered batch
            img_full[i:i + batch_size] = img_batch

            # Delete temporary tensors and free GPU memory
            del rgb, img_batch
            torch.cuda.empty_cache()
        
        # Reshape and convert the final image to uint8
        img = (img_full.reshape(h, w, 3) * 255).astype(np.uint8)
        frames.append(img)

        if i % 50 == 0:
          plt.imshow(img)
          plt.show()

        # Delete temporary tensors and free GPU memory
        del rays_o, rays_d
        torch.cuda.empty_cache()

    # Save the frames as an MP4 file
    output_file = RES_PATH + out_path
    fps = 30
    imageio.mimwrite(output_file, frames, fps=fps, quality=9, macro_block_size=None)

    print('Video saved to {}'.format(output_file))


def main(args):
    if not args.test:
        print('Training mode.')
        print("Trainging on device: ", DEVICE)
        # create a new model
        model = NeRF()

        # load the training data
        d = pickle.load(open(args.train_data, 'rb'))
        rays = d['rays']
        h, w, focal = d['h'], d['w'], d['focal']

        # load rays into a tensor
        training_dataset = torch.from_numpy(rays).float()

        print('Training data loaded.')
        print('Training dataset size: ', training_dataset.shape)
        print('Image size: {}x{}'.format(h, w))
        print('Focal length: ', focal)

        # load the testing data if provided
        if args.test_data is not None:
            d = pickle.load(open(args.test_data, 'rb'))
            test_rays = d['rays']
            testing_dataset = torch.from_numpy(test_rays).float()
        else:
            testing_dataset = None

        # train the model
        losses = do_training(model, training_dataset, testing_dataset,
                             num_ray_samples=args.samples, h=h, w=w)

        # save the model to the provided path
        torch.save(model.state_dict(), RES_PATH + 'model.pth')

        print('Model saved to: ', RES_PATH + 'model.pth')

        # show the training loss plot
        plt.plot(losses)
        plt.show()

    else:
        print('Testing mode.')
        # load the training data
        print('Loading data...')
        # load dict from pickle file
        d = pickle.load(open(args.test_data, 'rb'))
        h, w, focal = d['g'], d['w'], d['focal']

        print('Image size: {}x{}'.format(h, w))
        print('Focal length: ', focal)
        # load the model from the provided path in args
        model = load_model(args.model)
        print('Model loaded from: ', args.model)
        # render the video
        print('Rendering video...')
        render_video(model, h, w, focal)
        print('Video saved to: ', RES_PATH + 'video.mp4')
        # save the video to the provided path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # boolean flag to train or render
    parser.add_argument('--test', action='store_true')
    # path to the pickle file containing the training data
    parser.add_argument('--train_data', type=str, default=None)
    # path to the pickle file containing the testing data
    parser.add_argument('--test_data', type=str, default=None)
    # number of samples to use for each ray
    parser.add_argument('--samples', type=int, default=64)
    # path to the model file
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    main(args)

    # example call
    # python3 nerf.py --train_data data/imgs/drums/drums_400.pkl
    # python3 nerf.py --test --test_data data/imgs/drums/drums_400.pkl --model results/drums/model.pth
