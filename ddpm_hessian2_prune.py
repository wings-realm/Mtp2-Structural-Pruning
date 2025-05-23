from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning','taylor2'])

parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

args = parser.parse_args()

device = args.device
print(f"Using device: {device}")


batch_size = args.batch_size
dataset = args.dataset

def compute_taylor12_importance(
    model, dataloader, scheduler, device, 
    K=30, alpha=0.5, eps: float = 1e-8, hutchinson_samples: int = 1
):
    """
    Computes |g·w + α·h·w²| per‐parameter using:
    g = avg ∂L/∂w;
    h = Hutchinson-estimated avg diag(∂²L/∂w²).
    Then normalizes each param to zero‐mean, unit‐std.
    """
    grad_accum  = {n: torch.zeros_like(p, device=device)
                   for n, p in model.named_parameters() if p.requires_grad}
    hess_accum  = {n: torch.zeros_like(p, device=device)
                   for n, p in model.named_parameters() if p.requires_grad}

    loss_fn = torch.nn.MSELoss()
    model.train()

    for step, batch in enumerate(dataloader):
        if step >= K:
            break
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device)
        noise = torch.randn_like(imgs)

        model.zero_grad()
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (imgs.size(0),), device=device
        ).long()
        noisy = scheduler.add_noise(imgs, noise, timesteps)
        pred  = model(noisy, timesteps)["sample"]
        loss  = loss_fn(pred, noise)

        # First-order gradients
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)

        for (name, p), g in zip(model.named_parameters(), grads):
            if p.requires_grad:
                grad_accum[name] += g.detach()

        # Hutchinson's approximation of diagonal Hessian
        for _ in range(hutchinson_samples):
            # Sample Rademacher vector (±1)
            vs = [torch.randint_like(p, high=2, device=device) * 2 - 1 for p in model.parameters() if p.requires_grad]

            # Compute v^T (∇²L) v ≈ ∇(g·v)
            grad_dot_v = sum((torch.sum(g * v) for g, v in zip(grads, vs)))
            hvs = torch.autograd.grad(grad_dot_v, list(model.parameters()), retain_graph=True)

            for (name, p), hv, v in zip(model.named_parameters(), hvs, vs):
                if p.requires_grad and hv is not None:
                    hess_accum[name] += (hv * v).detach()

    # Average
    for name in grad_accum:
        grad_accum[name] /= K
        hess_accum[name] /= (K * hutchinson_samples)

    # Build normalized importance
    imp_dict = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        w = p.detach()
        g = grad_accum[name]
        h = hess_accum[name]

        imp = (g * w + alpha * h * w**2).abs()
        m, s = imp.mean(), imp.std()
        imp_norm = (imp - m) / (s + eps)
        imp_dict[name] = imp_norm

    return imp_dict


class Taylor12Importance(tp.importance.Importance):
    def __init__(self, importance_dict, model):
        # map each real Parameter → its precomputed importance tensor
        self.imp_map = {}
        for name, p in model.named_parameters():
            if name in importance_dict:
                self.imp_map[p] = importance_dict[name]

    def __call__(self, group, ch_groups=None):
        handler, _ = group[0]
        module = handler.target.module

        # fetch the exact Parameter this handler prunes
        try:
            p = getattr(module, handler.param_name)
        except AttributeError:
            # fallback: pick the first parameter we have a map for
            for cand in module.parameters(recurse=False):
                if cand in self.imp_map:
                    p = cand
                    break
            else:
                p = next(module.parameters())

        imp_tensor = self.imp_map.get(p, torch.zeros_like(p))

        # Only do channel‐level sums if ch_groups is actually a list/tuple
        if isinstance(ch_groups, (list, tuple)):
            per_chan = []
            for start, size in ch_groups:
                block = imp_tensor[start : start + size]
                per_chan.append(block.reshape(size, -1).sum(dim=1))
            return torch.cat(per_chan, dim=0)

        # otherwise just return the full per-weight importances
        return imp_tensor
    
    
if __name__=='__main__':
    
    # loading images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning','taylor2']:
        dataset = utils.get_dataset(args.dataset)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        import torch_pruning as tp
        clean_images = next(iter(train_dataloader))
        if isinstance(clean_images, (list, tuple)):
            clean_images = clean_images[0]
        clean_images = clean_images.to(args.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    if 'cifar' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}
    else:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio>0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
        elif args.pruner == 'taylor2':
            importance_dict = compute_taylor12_importance(
                model, train_dataloader, pipeline.scheduler, device=args.device, K=30
            )            
            
            imp = Taylor12Importance(importance_dict, model)
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}
        #from diffusers.models.attention import 
        #for m in model.modules():
        #    if isinstance(m, AttentionBlock):
        #        channel_groups[m.query] = m.num_heads
        #        channel_groups[m.key] = m.num_heads
        #        channel_groups[m.value] = m.num_heads
        
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            pruning_ratio=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()
        import random

        if args.pruner in ['taylor', 'diff-pruning','taylor2']:
            loss_max = 0
            print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(1000)):
                timesteps = (step_k*torch.ones((args.batch_size,), device=clean_images.device)).long()
                noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                model_output = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(model_output, noise) 
                loss.backward() 
                
                if args.pruner=='diff-pruning' or args.pruner=='taylor2':
                    if loss>loss_max: loss_max = loss
                    if loss<loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        from diffusers.models.resnet import Upsample2D, Downsample2D
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
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

    # Sampling images from the pruned model
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
        
