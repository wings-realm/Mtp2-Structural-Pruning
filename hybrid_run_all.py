import subprocess
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default="0", help="CUDA device number, e.g., 0 for cuda:0")
parser.add_argument("--pruning_ratio", type=float, default=0.3, help="Pruning ratio to be used in model pruning and evaluation")
args = parser.parse_args()

cuda_device = f"cuda:{args.cuda}"
pruning_ratio = args.pruning_ratio
pruning_ratio_str = str(pruning_ratio).replace('.', '_')  # for safe filenames like 0_3
run_prefix = f"run{pruning_ratio_str}/"

commands = [
    # {
    #     "desc": "Step 1: Pruning the pretrained DDPM model",
    #     "cmd": f"python3 ddpm_prune_hybrid.py --dataset cifar10 --model_path pretrained/ddpm_ema_cifar10 --save_path run/pruned/ddpm_cifar10_pruned --pruning_ratio {pruning_ratio} --batch_size 128 --pruner hybrid --lambda_weight 0.7 --device {{device}}"
    # },
    # {
    #     "desc": "Step 2: Fine-tuning the pruned model",
    #     "cmd": "python3 ddpm_train.py --dataset=cifar10 --model_path=run/pruned/ddpm_cifar10_pruned --pruned_model_ckpt=run/pruned/ddpm_cifar10_pruned/pruned/unet_pruned.pth --resolution=32 --output_dir=run/finetuned/ddpm_cifar10_pruned_post_training --train_batch_size=128 --num_iters=100000 --gradient_accumulation_steps=1 --learning_rate=2e-4 --lr_warmup_steps=0 --save_model_steps=1000 --dataloader_num_workers=8 --adam_weight_decay=0.00 --ema_max_decay=0.9999 --dropout=0.1 --use_ema --logging_dir=run/logs/ddpm_cifar10_pruned --device {device}"
    # },
    {
        "desc": "Step 3: Sampling from the fine-tuned pruned model",
        "cmd": "python3 ddpm_sample.py --output_dir run/sample/ddpm_cifar10_pruned --batch_size 128 --pruned_model_ckpt run/finetuned/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth --model_path run/finetuned/ddpm_cifar10_pruned_post_training --skip_type uniform --device {device}"
    },
    {
        "desc": "Step 4: Sampling from the pretrained model",
        "cmd": "python3 ddpm_sample.py --output_dir run/sample/ddpm_cifar10_pretrained --batch_size 128 --model_path pretrained/ddpm_ema_cifar10 --device {device}"
    },
    {
        "desc": "Step 5: Generating FID statistics from real CIFAR-10 images",
        "cmd": f"python3 fid_score.py --save-stats data/cifar10_images run/fid_stats_cifar10.npz run/sample/ddpm_cifar10_pretrained --device {{device}} --batch-size 256"
    },
    {
        "desc": "Step 6: Computing FID and SSIM score for pruned model samples",
        "cmd": f"python3 fid_score.py run/sample/ddpm_cifar10_pruned run/fid_stats_cifar10.npz run/sample/ddpm_cifar10_pretrained --device {{device}} --batch-size 256 --pruning_ratio {pruning_ratio_str}"
    }
]

for i, step in enumerate(commands, 1):
    print(f"\n===== {step['desc']} =====")
    # Replace all "run/" with "run<ratio>/" before formatting
    modified_cmd = step["cmd"].replace("run/", run_prefix)
    formatted_cmd = modified_cmd.format(device=cuda_device)
    print(f"Running command: {formatted_cmd}\n")
    result = subprocess.run(formatted_cmd, shell=True)
    if result.returncode != 0:
        print(f"Error occurred during Step {i}. Exiting.")
        break
    print(f"Completed Step {i}\n")
