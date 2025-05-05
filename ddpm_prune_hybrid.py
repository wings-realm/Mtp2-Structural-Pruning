from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning', 'hybrid'])
parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")
parser.add_argument("--lambda_weight", type=float, default=0.5, help="weighting factor between Taylor and latent scores")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
args.device = device

class HybridImportance(tp.importance.Importance):
    def __init__(self, hybrid_scores):
        super().__init__()
        self.hybrid_scores = hybrid_scores  # Dict[layer_name] = 1D tensor of scores per channel

    def __call__(self, group, *args, **kwargs):
        for dep, idxs in group:
            module = dep.target.module  # The actual nn.Module
            layer_name = dep.target.name  # Unique name assigned by torch-pruning

            if layer_name in self.hybrid_scores:
                # Ensure the tensor is on the same device
                scores = self.hybrid_scores[layer_name].to(next(module.parameters()).device)
                return scores[idxs]  # Return the score for the pruned indices

        # Fallback: return dummy importance
        fallback_len = len(group[0][1])
        device = next(group[0][0].target.module.parameters()).device
        return torch.ones(fallback_len, device=device)


def is_channel_weight(param):
    return param.ndim >= 2  # e.g., conv weights
    
def compute_latent_sensitivity(model, sampler, num_samples=10, alpha=0.1, batch_size=16):
    latent_scores = {}
    for name, param in model.named_parameters():
        if is_channel_weight(param):
            latent_scores[name] = torch.zeros(param.shape[0], device=param.device)
    model.eval()
    for _ in range(num_samples):
        z = torch.randn((batch_size, 3, 32, 32), device=args.device)
        d = torch.randn_like(z)
        z_pert = z + alpha * d

        x = sampler(z)
        x_pert = sampler(z_pert)
        loss = F.l1_loss(x, x_pert)
        model.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        del x, x_pert, loss

        for name, param in model.named_parameters():
            if is_channel_weight(param) and param.grad is not None:
                grad_mag = param.grad.abs().reshape(param.shape[0], -1).sum(dim=1)
                latent_scores[name] += grad_mag.detach()

    for name in latent_scores:
        latent_scores[name] /= float(num_samples)
    return latent_scores

if __name__=='__main__':
    if args.pruner in ['taylor', 'diff-pruning', 'hybrid']:
        dataset = utils.get_dataset(args.dataset)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        clean_images = next(iter(train_dataloader))
        if isinstance(clean_images, (list, tuple)):
            clean_images = clean_images[0]
        clean_images = clean_images.to(args.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)

    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio > 0:
        if args.pruner in ['taylor', 'diff-pruning', 'hybrid']:
            imp = tp.importance.TaylorImportance(multivariable=args.pruner != 'diff-pruning')
        elif args.pruner == 'random' or args.pruner == 'reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}


        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        loss_max = 0
        print("Accumulating gradients for pruning...")
        for step_k in tqdm(range(1000)):
            model.zero_grad()
            timesteps = (step_k * torch.ones((args.batch_size,), device=clean_images.device)).long()
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            model_output = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(model_output, noise)
            loss.backward()
            loss_max = max(loss_max, loss.item())

            if args.pruner == 'diff-pruning':
                if loss > loss_max: loss_max = loss
                if loss < loss_max * args.thr: break

    
        if args.pruner == 'hybrid':
            print("Computing latent sensitivity...")

            def sampler(z):
                scheduler.set_timesteps(10)
                z = z.to(args.device)
                for t in scheduler.timesteps:
                    noise_pred = model(z, t).sample  # no torch.no_grad() here
                    z = scheduler.step(noise_pred, t, z).prev_sample
                return z

            latent_scores = compute_latent_sensitivity(model, sampler, num_samples=10, alpha=0.1, batch_size=args.batch_size)
            
            taylor_scores = {}
            for name, param in model.named_parameters():
                if is_channel_weight(param) and param.grad is not None:
                    taylor_scores[name] = (param * param.grad).reshape(param.shape[0], -1).sum(dim=1).abs()

            hybrid_scores = {}
            for name in taylor_scores:
                ts = taylor_scores[name] / (taylor_scores[name].max() + 1e-8)
                ls = latent_scores[name] / (latent_scores[name].max() + 1e-8)
                hybrid_scores[name] = args.lambda_weight * ts + (1 - args.lambda_weight) * ls
                imp2 = HybridImportance(hybrid_scores) # override score used by pruner
            imp = imp2
        
        pruner.importance = imp

        for g in pruner.step(interactive=True):
            g.prune()

        from diffusers.models.resnet import Upsample2D, Downsample2D
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels = m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        # print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        model.zero_grad()
        del pruner

        if args.pruner=='reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            reset_parameters(model)

    pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio>0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))

    pipeline = DDIMPipeline(
        unet = model,
        scheduler = DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        pipeline.to("cuda")
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator, output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(args.save_path))
