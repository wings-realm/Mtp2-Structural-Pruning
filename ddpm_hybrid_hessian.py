# ddpm_prune_hybrid12.py
import torch
import torch.nn.functional as F
import torch_pruning as tp
import torchvision
from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler
import argparse, os
from tqdm import tqdm
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--lambda_taylor", type=float, default=0.4)
parser.add_argument("--lambda_latent", type=float, default=0.3)
parser.add_argument("--lambda_hessian", type=float, default=0.3)
args = parser.parse_args()

device = torch.device(args.device)
print(f"Using device: {device}")

def is_channel_weight(param):
    return param.ndim >= 2

def compute_latent_sensitivity(model, sampler, alpha=0.1, num_samples=10, batch_size=16):
    scores = {}
    for name, p in model.named_parameters():
        if is_channel_weight(p):
            scores[name] = torch.zeros(p.shape[0], device=p.device)
    model.eval()
    for _ in range(num_samples):
        z = torch.randn((batch_size, 3, 32, 32), device=device)
        d = torch.randn_like(z)
        z_pert = z + alpha * d
        x = sampler(z)
        x_pert = sampler(z_pert)
        loss = F.l1_loss(x, x_pert)
        model.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        del x, x_pert, loss

        for name, p in model.named_parameters():
            if is_channel_weight(p) and p.grad is not None:
                grad_mag = p.grad.abs().reshape(p.shape[0], -1).sum(dim=1)
                scores[name] += grad_mag.detach()
        
    for k in scores:
        scores[k] /= float(num_samples)
    return scores

def compute_hessian_diag(model, dataloader, scheduler, K=60, alpha=0.5, eps=1e-8, hutchinson_samples=1):
    grad_accum = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    hess_accum = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    loss_fn = torch.nn.MSELoss()
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= K: break
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        noise = torch.randn_like(x)
        model.zero_grad()
        t = torch.randint(0, scheduler.config.num_train_timesteps, (x.size(0),), device=device).long()
        noisy = scheduler.add_noise(x, noise, t)
        pred = model(noisy, t).sample
        loss = loss_fn(pred, noise)
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        for (name, p), g in zip(model.named_parameters(), grads):
            if p.requires_grad:
                grad_accum[name] += g.detach()
        for _ in range(hutchinson_samples):
            vs = [torch.randint_like(p, 2, device=device) * 2 - 1 for p in model.parameters() if p.requires_grad]
            grad_dot_v = sum(torch.sum(g * v) for g, v in zip(grads, vs))
            hvs = torch.autograd.grad(grad_dot_v, list(model.parameters()), retain_graph=True)
            for (name, p), hv, v in zip(model.named_parameters(), hvs, vs):
                if p.requires_grad and hv is not None:
                    hess_accum[name] += (hv * v).detach()
        

    for name in grad_accum:
        grad_accum[name] /= K
        hess_accum[name] /= (K * hutchinson_samples)
    imp = {}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        w, g, h = p.detach(), grad_accum[name], hess_accum[name]
        score = (g * w + alpha * h * w**2).abs()
        m, s = score.mean(), score.std()
        imp[name] = (score - m) / (s + eps)
    return imp

class Hybrid123Importance(tp.importance.Importance):
    def __init__(self, taylor, latent, hessian, lambdas):
        self.taylor = taylor
        self.latent = latent
        self.hessian = hessian
        self.l1, self.l2, self.l3 = lambdas
    def __call__(self, group, *args, **kwargs):
        for handler, idxs in group:
            module = handler.target.module
            name = handler.target.name
            device = next(module.parameters()).device
            t = self.taylor.get(name, torch.zeros(len(idxs), device=device))
            l = self.latent.get(name, torch.zeros(len(idxs), device=device))
            h = self.hessian.get(name, torch.zeros(len(idxs), device=device))
            t = t / (t.max() + 1e-8)
            l = l / (l.max() + 1e-8)
            h = h / (h.max() + 1e-8)
            return self.l1 * t[idxs] + self.l2 * l[idxs] + self.l3 * h[idxs]

if __name__ == "__main__":
    dataset = utils.get_dataset(args.dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    clean_images = next(iter(train_loader))
    clean_images = clean_images[0] if isinstance(clean_images, (list, tuple)) else clean_images
    clean_images = clean_images.to(device)
    noise = torch.randn_like(clean_images)

    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(device), 'timestep': torch.ones((1,), device=device).long()}

    # Gradient Accumulation
    print("Accumulating gradients for pruning...")
    model.zero_grad()
    for step in tqdm(range(1000)):
        timesteps = (step * torch.ones((args.batch_size,), device=device)).long()
        noisy = scheduler.add_noise(clean_images, noise, timesteps)
        out = model(noisy, timesteps).sample
        loss = F.mse_loss(out, noise)
        loss.backward()


    # Compute importance scores
    print("Computing Taylor/Latent/Hessian importance...")
    taylor_scores = {n: (p * p.grad).reshape(p.shape[0], -1).sum(1).abs() for n, p in model.named_parameters() if is_channel_weight(p) and p.grad is not None}
    def sampler(z):
        scheduler.set_timesteps(10)
        z = z.to(args.device)
        for t in scheduler.timesteps:
            noise_pred = model(z, t).sample  # no torch.no_grad() here
            z = scheduler.step(noise_pred, t, z).prev_sample
        return z

    latent_scores = compute_latent_sensitivity(model, sampler, batch_size=args.batch_size)
    hessian_scores = compute_hessian_diag(model, train_loader, scheduler)
    
    latent_scores = (latent_scores - min(latent_scores)) / (max(latent_scores) - min(latent_scores))
    hessian_scores = (hessian_scores - min(hessian_scores)) / (max(hessian_scores) - min(hessian_scores))
    taylor_scores = (taylor_scores - min(taylor_scores)) / (max(taylor_scores) - min(taylor_scores))
    
    # Create hybrid importance
    importance = Hybrid123Importance(taylor_scores, latent_scores, hessian_scores, 
                                     lambdas=(args.lambda_taylor, args.lambda_latent, args.lambda_hessian))

    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs, importance=importance,
        iterative_steps=1, ch_sparsity=args.pruning_ratio,
        ignored_layers=[model.conv_out], channel_groups={}
    )

    for g in pruner.step(interactive=True):
        g.prune()

    # Fix residual modules
    from diffusers.models.resnet import Upsample2D, Downsample2D
    for m in model.modules():
        if isinstance(m, (Upsample2D, Downsample2D)):
            m.channels = m.conv.in_channels
            m.out_channels = m.conv.out_channels

    pipeline.save_pretrained(args.save_path)
    os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
    torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))

    # Generate images
    pipeline = DDIMPipeline(unet=model, scheduler=DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")).to(device)
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator, output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), f"{args.save_path}/vis/after_pruning.png")
