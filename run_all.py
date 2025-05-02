import subprocess

commands = [
    {
        "desc": "Step 1: Pruning the pretrained DDPM model",
        "cmd": "python ddpm_prune.py --dataset cifar10 --model_path pretrained/ddpm_ema_cifar10 --save_path run/pruned/ddpm_cifar10_pruned --pruning_ratio 0.3 --batch_size 128 --pruner diff-pruning --thr 0.05 --device cuda:0"
    },
    {
        "desc": "Step 2: Fine-tuning the pruned model",
        "cmd": "python ddpm_train.py --dataset=cifar10 --model_path=run/pruned/ddpm_cifar10_pruned --pruned_model_ckpt=run/pruned/ddpm_cifar10_pruned/pruned/unet_pruned.pth --resolution=32 --output_dir=run/finetuned/ddpm_cifar10_pruned_post_training --train_batch_size=128 --num_iters=5000 --gradient_accumulation_steps=1 --learning_rate=2e-4 --lr_warmup_steps=0 --save_model_steps=1000 --dataloader_num_workers=8 --adam_weight_decay=0.00 --ema_max_decay=0.9999 --dropout=0.1 --use_ema --logging_dir=run/logs/ddpm_cifar10_pruned"
    },
    {
        "desc": "Step 3: Sampling from the fine-tuned pruned model",
        "cmd": "python ddpm_sample.py --output_dir run/sample/ddpm_cifar10_pruned --batch_size 128 --pruned_model_ckpt run/finetuned/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth --model_path run/finetuned/ddpm_cifar10_pruned_post_training --skip_type uniform"
    },
    # {
    #     "desc": "Step 4: Sampling from the pretrained model",
    #     "cmd": "python ddpm_sample.py --output_dir run/sample/ddpm_cifar10_pretrained --batch_size 128 --model_path pretrained/ddpm_ema_cifar10"
    # },
    {
        "desc": "Step 5: Generating FID statistics from real CIFAR-10 images",
        "cmd": "python fid_score.py --save-stats data/cifar10_images run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256"
    },
    {
        "desc": "Step 6: Computing FID score for pruned model samples",
        "cmd": "python fid_score.py run/sample/ddpm_cifar10_pruned run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256"
    }
]

for i, step in enumerate(commands, 1):
    print(f"\n===== {step['desc']} =====")
    print(f"Running command: {step['cmd']}\n")
    result = subprocess.run(step['cmd'], shell=True)
    if result.returncode != 0:
        print(f"Error occurred during Step {i}. Exiting.")
        break
    print(f"Completed Step {i}\n")
