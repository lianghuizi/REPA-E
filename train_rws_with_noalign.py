import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import wandb

# from dataset import CustomINH5Dataset
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from models.sit import SiT_models
from models.rae import RAE                                  # 添加
from samplers import euler_sampler
from utils import load_encoders, normalize_latents, denormalize_latents, preprocess_imgs_vae, count_trainable_params

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    # 我们不再使用 x.shape[-1] 去做危险的除法计算
    # 直接固定目标尺寸为 224，这是 DINOv2/CLIP 的原生最佳尺寸
    # 综合算力换成112
    target_size = (112, 112) 

    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, size=target_size, mode='bicubic', align_corners=False)

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale # normalize
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    # if args.vae == "f8d4":
    #     assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    #     latent_size = args.resolution // 8
    #     in_channels = 4
    # elif args.vae == "f16d32":
    #     assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
    #     latent_size = args.resolution // 16
    #     in_channels = 32
    # else:
    #     raise NotImplementedError()
    latent_size = 8
    in_channels = 768

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        **block_kwargs
    )

    # make a copy of the model for EMA
    model = model.to(device)
    # ema = copy.deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Load VAE and create a EMA of the VAE
    # vae = vae_models[args.vae]().to(device)
    # 实例化 RAE
    # 实例化 RAE
    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base', 
        encoder_params={
        "dinov2_path": "facebook/dinov2-base"   # ✅ 关键
        },
        encoder_input_size=112,                      # 依据你 train.py 中 preprocess_raw_image 的 target_size
        decoder_config_path='facebook/vit-mae-base',
        decoder_patch_size=16,
        noise_tau=0.8,                               # 联合训练必须保留加噪，构建连续的潜在空间
        reshape_to_2d=True,                          # SiT 需要二维空间特征图
        # 如果你之前跑通过 RAE 并提取了统计信息，可以传入路径进行 Z-score 归一化；否则设为 None
        normalization_stat_path=None                 
    ).to(device)

    # 加载 RAE 权重
    rae_ckpt = torch.load(args.rae_ckpt, map_location=device)
    rae.load_state_dict(rae_ckpt, strict=False)
    # vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    # vae.load_state_dict(vae_ckpt, strict=False)   # We may not have the projection layer in the VAE
    del rae_ckpt
    # requires_grad(ema, False)
    print("Total trainable params in RAE:", count_trainable_params(rae))

    # Load the VAE latents-stats for BN layer initialization
    latents_stats = torch.load(args.rae_ckpt.replace(".pt", "-latents-stats.pt"))
    # latents_scale = latents_stats["latents_scale"].squeeze().to(device)
    # latents_bias = latents_stats["latents_bias"].squeeze().to(device)
    # 适配 'mean' 和 'var' 键名
    # 计算标准差作为 scale: std = sqrt(var)
    # mean=None 说明均值为0，直接用零向量
    if latents_stats["mean"] is None:
        latents_bias = torch.zeros(in_channels).to(device)
    else:
        latents_bias = latents_stats["mean"].squeeze().to(device)
    # 修复
    var = latents_stats["var"]  # [768, 8, 8]
    if var.dim() == 3:
        var = var.mean(dim=[1, 2])  # → [768]
    latents_scale = torch.sqrt(var + 1e-6).to(device)

    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)
    
    ema = copy.deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Apply SyncBN if more than 1 GPU is used
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    loss_cfg = OmegaConf.load("configs/l1_lpips_rae_gan.yaml")
    rae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for SiT, VAE, and VAE loss function separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_rae = torch.optim.AdamW(
        rae.parameters(),
        lr=args.rae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_loss_fn = torch.optim.AdamW(
        rae_loss_fn.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data
    # Tiny-ImageNet 均值和标准差 (ImageNet 标准值通常也适用)
    tiny_mean = (0.4802, 0.4481, 0.3975)
    tiny_std = (0.2302, 0.2265, 0.2262)
    
    transform = transforms.Compose([
        # 考虑到你模型要求分辨率（通常是256），先Resize再Crop
        # 这里是128
        transforms.Resize(args.resolution + 32), 
        transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Tiny-ImageNet 结构通常是 root/train/class_id/*.JPEG
    train_dir = os.path.join(args.data_dir, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"未在 {train_dir} 找到数据，请检查 Tiny-ImageNet 路径")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # Start with eval mode for all models
    model.eval()
    ema.eval()
    rae.eval()

    if args.disc_pretrained_ckpt is not None:
        # Load the discriminator from a pretrained checkpoint if provided
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        rae_loss_fn.discriminator.load_state_dict(disc_ckpt)
        if accelerator.is_main_process:
            logger.info(f"Loaded discriminator from {args.disc_pretrained_ckpt}")

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.continue_train_exp_dir}/checkpoints/{ckpt_name}'

        # If the checkpoint exists, we load the checkpoinrt and resume the training
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        rae.load_state_dict(ckpt['rae'])
        rae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])
        optimizer.load_state_dict(ckpt['opt']),
        optimizer_rae.load_state_dict(ckpt['opt_rae'])
        optimizer_loss_fn.load_state_dict(ckpt['opt_disc'])
        global_step = ckpt['steps']

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")
        rae = torch.compile(rae, backend="inductor", mode="default")
        rae_loss_fn = torch.compile(rae_loss_fn, backend="inductor", mode="default")

    model, rae, rae_loss_fn, optimizer, optimizer_rae, optimizer_loss_fn, train_dataloader = accelerator.prepare(
        model, rae, rae_loss_fn, optimizer, optimizer_rae, optimizer_loss_fn, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="gradient-pass-through",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 8 // accelerator.num_processes
    ys = torch.randint(200, size=(sample_batch_size,), device=device)     # 1000----->200
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)

    # main training loop
    for epoch in range(args.epochs):
        model.train()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)
            z = None

            # # extract the dinov2 features
            # with torch.no_grad():
            #     zs = []
            #     with accelerator.autocast():
            #         for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
            #             raw_image_ = preprocess_raw_image(raw_image, encoder_type)
            #             z = encoder.forward_features(raw_image_)
            #             if 'mocov3' in encoder_type: z = z = z[:, 1:] 
            #             if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
            #             zs.append(z)
            zs = None #策略C不需要教师特征

            rae.train()
            model.train()
            with accelerator.accumulate([model, rae, rae_loss_fn]), accelerator.autocast():
                # posterior, z, recon_image = vae(processed_image) 去掉
                # --- 修改后 (RAE 逻辑) ---
                processed_image = raw_image
                z = rae.encode(processed_image)
                recon_image = rae.decode(z)

                 # 2. 为计算损失准备 [-1.0, 1.0] 的数据版本
                # 公式: x * 2.0 - 1.0
                raw_image_for_loss = raw_image * 2.0 - 1.0
                recon_image_for_loss = recon_image * 2.0 - 1.0
                
                # 3. 统一尺寸：把 target 缩放到和 recon 一致
                if raw_image_for_loss.shape[-1] != recon_image_for_loss.shape[-1]:
                    processed_image_for_loss = torch.nn.functional.interpolate(
                    raw_image_for_loss, 
                    size=recon_image_for_loss.shape[-2:],  
                    mode='bilinear', 
                    align_corners=False
                )
                else:
                    processed_image_for_loss = raw_image_for_loss

                # 2). Backward pass: VAE, compute the VAE loss, backpropagate, and update the VAE; Then, compute the discriminator loss and update the discriminator
                #    loss_kwargs used for SiT forward function, create here and can be reused for both VAE and SiT
                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )
                # Record the time_input and noises for the VAE alignment, so that we avoid sampling again
                time_input = None
                noises = None

                # Turn off grads for the SiT model (avoid REPA gradient on the SiT model)
                requires_grad(model, False)
                # Avoid BN stats to be updated by the VAE
                model.eval()

                # vae_loss, vae_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "generator")         去掉
                # 因为没有 posterior，调用损失函数时传 None 或修改损失类内部逻辑
                # 如果损失函数里强制要 posterior 算 KL，你需要把 KL 权重设为 0
                # --- 修改后 ---
                # --- 动态获取 RAE Decoder 的最后一层权重 ---
                last_layer_weight = None
                # 倒序遍历 decoder 的所有参数
                for name, param in reversed(list(rae.decoder.named_parameters())):
                    # 找到最后一个包含 "weight" 且参与梯度的参数（跳过 bias 和 layernorm 的缩放因子）
                    if "weight" in name and param.requires_grad and len(param.shape) >= 2:
                        last_layer_weight = param
                        break

                if last_layer_weight is None:
                    raise ValueError("无法在 RAE Decoder 中找到有效的最后一层权重矩阵！")

                # # Compute the REPA alignment loss for VAE updates
                # loss_kwargs["align_only"] = True
                # rae_align_outputs = model(
                #     x=z,
                #     y=labels,
                #     zs=zs,
                #     loss_kwargs=loss_kwargs,
                #     time_input=time_input,
                #     noises=noises,
                # )
                
                # # 3. 计算 RAE Loss
                # extra_dict = {
                #     "zs": zs, 
                #     "zs_tilde": rae_align_outputs["zs_tilde"], 
                #     "last_layer": last_layer_weight
                # }
                # rae_loss, rae_loss_dict = rae_loss_fn(processed_image, recon_image, extra_dict, global_step, "generator")
                # rae_loss = rae_loss.mean() + args.rae_align_proj_coeff * rae_align_outputs["proj_loss"].mean()
                
                # # Save the `time_input` and `noises` and reuse them for the SiT model forward pass
                # time_input = rae_align_outputs["time_input"]
                # noises = rae_align_outputs["noises"]

                # 计算纯粹的 RAE Loss (仅包含重构、感知与 GAN，如果需要自适应权重只需传入 last_layer)
                extra_dict = {
                    "last_layer": last_layer_weight
                }

                rae_loss, rae_loss_dict = rae_loss_fn(processed_image_for_loss, recon_image_for_loss, extra_dict, global_step, "generator")
                rae_loss = rae_loss.mean()
                
                accelerator.backward(rae_loss)
                if accelerator.sync_gradients:
                    grad_norm_rae = accelerator.clip_grad_norm_(rae.parameters(), args.max_grad_norm)
                optimizer_rae.step()
                optimizer_rae.zero_grad(set_to_none=True)

                # discriminator loss and update
                d_loss, d_loss_dict = rae_loss_fn(processed_image_for_loss, recon_image_for_loss, extra_dict, global_step, "discriminator")
                d_loss = d_loss.mean()
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(rae_loss_fn.parameters(), args.max_grad_norm)
                optimizer_loss_fn.step()
                optimizer_loss_fn.zero_grad(set_to_none=True)

                # Turn the grads back on for the SiT model, and put the model into training mode
                requires_grad(model, True)
                model.train()

                # 3). Forward pass: SiT
                # **Avoid diffusion loss to backpropagate to the VAE, so we detach the `z`**
                loss_kwargs["weighting"] = args.weighting
                loss_kwargs["align_only"] = False
                sit_outputs = model(
                    x=z.detach(),
                    y=labels,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_input,
                    noises=noises,
                )

                # 4). Compute diffusion loss and REPA alignment loss, backpropagate the SiT loss, and update the model
                sit_loss = sit_outputs["denoising_loss"].mean() 
                # 去掉这行：+ args.proj_coeff * sit_outputs["proj_loss"].mean()
                accelerator.backward(sit_loss)
                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # 5). Update SiT EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(ema, unwrapped_model._orig_mod if args.compile else unwrapped_model)

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(), 
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                    # 删除这行：
                    #"proj_loss": accelerator.gather(sit_outputs["proj_loss"]).mean().detach().item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    "epoch": epoch,
                    "rae_loss": accelerator.gather(rae_loss).mean().detach().item(),
                    "reconstruction_loss": accelerator.gather(rae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                    "perceptual_loss": accelerator.gather(rae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                    #"kl_loss": accelerator.gather(rae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                    "weighted_gan_loss": accelerator.gather(rae_loss_dict["weighted_gan_loss"].mean()).mean().detach().item(),
                    "discriminator_factor": accelerator.gather(rae_loss_dict["discriminator_factor"].mean()).mean().detach().item(),
                    "gan_loss": accelerator.gather(rae_loss_dict["gan_loss"].mean()).mean().detach().item(),
                    "d_weight": accelerator.gather(rae_loss_dict["d_weight"].mean()).mean().detach().item(),
                    "grad_norm_rae": accelerator.gather(grad_norm_rae).mean().detach().item(),
                    #"rae_align_loss": accelerator.gather(rae_align_outputs["proj_loss"].mean()).mean().detach().item(),
                    "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                    "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().detach().item(),
                    "logits_real": accelerator.gather(d_loss_dict["logits_real"].mean()).mean().detach().item(),
                    "logits_fake": accelerator.gather(d_loss_dict["logits_fake"].mean()).mean().detach().item(),
                    "lecam_loss": accelerator.gather(d_loss_dict["lecam_loss"].mean()).mean().detach().item(),
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` and `vae` are wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_rae = accelerator.unwrap_model(rae)
                    unwrapped_rae_loss_fn = accelerator.unwrap_model(rae_loss_fn)

                    # model might be compiled, we extract the original model
                    original_model = unwrapped_model._orig_mod if args.compile else unwrapped_model
                    original_rae = unwrapped_rae._orig_mod if args.compile else unwrapped_rae
                    original_discriminator = unwrapped_rae_loss_fn._orig_mod.discriminator if args.compile else unwrapped_rae_loss_fn.discriminator

                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "rae": original_rae.state_dict(),
                        "discriminator": original_discriminator.state_dict(),
                        "opt": optimizer.state_dict(),
                        "opt_rae": optimizer_rae.state_dict(),
                        "opt_disc": optimizer_loss_fn.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                # NOTE: Inference should use eval mode
                model.eval()
                rae.eval()
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_sampler(
                        unwrapped_model,
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    latents_stats = unwrapped_model.extract_latents_stats()
                    # reshape latents_stats to [1, C, 1, 1]
                    latents_scale = latents_stats['latents_scale'].view(1, in_channels, 1, 1)
                    latents_bias = latents_stats['latents_bias'].view(1, in_channels, 1, 1)
                    samples = accelerator.unwrap_model(rae).decode(
                        denormalize_latents(samples, latents_scale, latents_bias))
                    
                out_samples = accelerator.gather(samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples))})
                logging.info("Generating EMA samples done.")
                torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=5000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT model params
    parser.add_argument("--model", type=str, default="SiT-B/2", choices=SiT_models.keys(),
                        help="The model to train.")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset params
    parser.add_argument("--data-dir", type=str, default="/mnt/workspace/data/tiny-imagenet-200")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # loss params
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"],
                        help="currently we only support v-prediction")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, choices=["uniform", "lognormal"],
                        help="Loss weihgting, uniform or lognormal")

    # vae params
    parser.add_argument("--rae", type=str, default="f8d4", choices=["f8d4", "f16d32"])
    parser.add_argument("--rae-ckpt", type=str, default="pretrained/sdvae-f8d4/sdvae-f8d4.pt")

    # vae loss params
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None)
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_kl_gan.yaml")

    # vae training params
    parser.add_argument("--rae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rae-align-proj-coeff", type=float, default=1.5)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
