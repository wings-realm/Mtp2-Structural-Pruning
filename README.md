# 🧠 Diffusion Model Pruning on CIFAR-10

This repository demonstrates structured pruning and fine-tuning on a pretrained DDPM (Denoising Diffusion Probabilistic Model) using the CIFAR-10 dataset. It also includes scripts for sampling and evaluating FID scores.

---

### 0. Requirements, Data and Pretrained Model

* Requirements
```bash
pip install -r requirements2.txt
```
 
* Data
  
Download and extract CIFAR-10 images to *data/cifar10_images* for training and evaluation.
```bash
python tools/extract_cifar10.py --output data
```
* Pretrained Models
  
The following script will download an official DDPM model and convert it to the format of Huggingface Diffusers. You can find the converted model at *pretrained/ddpm_ema_cifar10*. It is an EMA version of [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32)
```bash
bash tools/convert_cifar10_ddpm_ema.sh
```

(Optional) You can also download a pre-converted model using wget
```bash
wget https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip
```

## 🔧 1. Pruning

Prune a pretrained DDPM model using Diff-Pruning with a given pruning ratio and threshold.

```bash
python ddpm_prune.py \
    --dataset cifar10 \
    --model_path pretrained/ddpm_ema_cifar10 \
    --save_path run/pruned/ddpm_cifar10_pruned \
    --pruning_ratio 0.3 \
    --batch_size 128 \
    --pruner diff-pruning \
    --thr 0.05 \
    --device cuda:0
```

## 🔧 2. Fine-Tuning

Fine-tune the pruned model to recover performance and quality.

```bash
python ddpm_train.py \
    --dataset cifar10 \
    --model_path run/pruned/ddpm_cifar10_pruned \
    --pruned_model_ckpt run/pruned/ddpm_cifar10_pruned/pruned/unet_pruned.pth \
    --resolution 32 \
    --output_dir run/finetuned/ddpm_cifar10_pruned_post_training \
    --train_batch_size 128 \
    --num_iters 100000 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lr_warmup_steps 0 \
    --save_model_steps 1000 \
    --dataloader_num_workers 8 \
    --adam_weight_decay 0.00 \
    --ema_max_decay 0.9999 \
    --dropout 0.1 \
    --use_ema \
    --logging_dir run/logs/ddpm_cifar10_pruned
```

## 🔧 3. Sampling from Pruned Model

Generate samples using the fine-tuned, pruned DDPM model.

```bash
python ddpm_sample.py \
    --output_dir run/sample/ddpm_cifar10_pruned \
    --batch_size 128 \
    --pruned_model_ckpt run/finetuned/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth \
    --model_path run/finetuned/ddpm_cifar10_pruned_post_training \
    --skip_type uniform
```

## 🔧 4. Sampling from Pretrained Model

Generate samples using the original pretrained DDPM model.

```bash
python ddpm_sample.py \
    --output_dir run/sample/ddpm_cifar10_pretrained \
    --batch_size 128 \
    --model_path pretrained/ddpm_ema_cifar10
```

## 🔧 5. Generate FID Stats (Reference)

Create FID statistics from real CIFAR-10 images.

```bash
python fid_score.py \
    --save-stats data/cifar10_images \
    run/fid_stats_cifar10.npz \
    --device cuda:0 \
    --batch-size 256
```

## 🔧 6. Compute FID of Pruned Model

Evaluate the sample quality of the pruned model using the FID score.

```bash
python fid_score.py \
    run/sample/ddpm_cifar10_pruned \
    run/fid_stats_cifar10.npz \
    --device cuda:0 \
    --batch-size 256
```

## 🔧 Merged Command

There are files such as for only hybrid running-> ddpm_prune_hybrid.py, you can run it by hybrid_run_all.py
ddpm_hessian2_prune.py, you can run it by hessian_run_all.py
ddpm_hybrid_hessian.py, you can run it by hybrid_hessian_run_all.py
