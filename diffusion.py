import argparse
from pathlib import Path
from math import ceil, sqrt
from typing import Optional

import torch
import wandb
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as tvtf
from torchvision.datasets import FashionMNIST
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import (
    DDPMPipeline,
    DDPMScheduler,
    UNet2DModel,
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
)

OUTPUT_IMAGES_PATH = Path("diffusion_imgs")
OUTPUT_IMAGES_PATH.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Subset:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


def custom_ddpm(
    notes: Optional[str] = None, debug: bool = False, skip_training: bool = False
):
    wandb.init(
        project="diffusion-customer-ddpm",
        config=dict(
            # Diffusion params
            T=1_000,  # like in the paper.
            beta_1=1e-4,
            beta_T=0.02,
            # Dataset params
            dataset_root=Path("data") / "fashion_mnist",
            use_dataset_subset=True,
            # Network architecture params
            model_block_out_channels=(32, 64, 128, 128),
            # Training params
            learning_rate=2e-3,
            n_epochs=1000,
            n_iter_per_epoch=None,
            skip_training=skip_training,
            batch_size=4,
            # Misc params
            seed=3,
            device=device,
            tags=("debug") if debug else None,
        ),
        notes=notes,
    )

    # Initialization
    torch.manual_seed(wandb.config.seed)

    # Dataset loading
    full_dataset = FashionMNIST(
        wandb.config.dataset_root,
        download=not Path(wandb.config.dataset_root).exists(),
        transform=tvtf.Compose(
            [
                tvtf.ToImage(),
                tvtf.ToDtype(torch.float32, scale=True),
                tvtf.Resize((32, 32), antialias=True),
                lambda x: (x * 2.0) - 1.0,
            ]
        ),
    )
    dataset = (
        full_dataset
        if not wandb.config.use_dataset_subset
        else Subset(full_dataset, n=wandb.config.batch_size)
    )

    # Model loading
    sample_image, sample_label = dataset[0]
    C, H, W = sample_image.shape
    model = UNet2DModel(
        sample_size=(H, W),
        in_channels=C,
        out_channels=C,
        block_out_channels=wandb.config.model_block_out_channels,
    ).to(device)

    # Initializing the DDPM constants that are used for forward and reverses processes.
    with torch.no_grad():
        beta = torch.linspace(
            wandb.config.beta_1, wandb.config.beta_T, steps=wandb.config.T
        ).to(device)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

    # Training loop (i.e. forward process).
    if not wandb.config.skip_training:
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=wandb.config.learning_rate,
            total_steps=ceil(len(dataset) / wandb.config.batch_size)
            * wandb.config.n_epochs,
        )

        for epoch in tqdm(range(wandb.config.n_epochs)):
            for i, (x0s, _) in enumerate(
                DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)
            ):
                if (
                    wandb.config.n_iter_per_epoch is not None
                    and i >= wandb.config.n_iter_per_epoch
                ):
                    break

                B, C, H, W = x0s.shape
                ts = torch.randint(
                    low=0,
                    high=wandb.config.T - 1,
                    size=(B,),
                    dtype=torch.long,
                    device=device,
                )
                x0s = x0s.to(device)

                noises = torch.randn_like(x0s, device=device)
                xtp1s = (
                    torch.sqrt(alpha_bar[ts]).view(B, 1, 1, 1) * x0s
                    + torch.sqrt(1 - alpha_bar[ts]).view(B, 1, 1, 1) * noises
                )

                predicted_noises = model(xtp1s, ts).sample
                loss = loss_fn(predicted_noises, noises)

                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

        torch.save(model.cpu().state_dict(), "/tmp/unet.bin")

    # Reverse process
    model.load_state_dict(torch.load("/tmp/unet.bin"))
    model = model.to(device).eval()
    batch_size = wandb.config.batch_size

    with torch.no_grad():
        xts = torch.randn((batch_size, C, H, W)).to(device)
        for t in tqdm(reversed(range(wandb.config.T))):
            ts = torch.full((batch_size, 1, 1, 1), t, device=device, dtype=torch.long)
            z = torch.randn_like(xts) if t > 0 else torch.zeros_like(xts)

            _alpha = alpha[ts]
            _beta = beta[ts]
            _alpha_bar = alpha_bar[ts]
            _sigma = torch.sqrt(_beta)

            estimated_noise = model(xts, ts.view(batch_size)).sample

            scaling_factor = 1 / torch.sqrt(_alpha)
            noise_scaling_factor = (1 - _alpha) / torch.sqrt(1 - _alpha_bar)
            mu = scaling_factor * (xts - noise_scaling_factor * estimated_noise)

            xts = mu + _sigma * z

            if t % 250 == 0 or (t < 50 and t % 10 == 0) or t == 0:
                torch_imgs_to_pil(xts).save(
                    OUTPUT_IMAGES_PATH / f"reverse_pass_{t:04d}.jpg"
                )


def torch_imgs_to_pil(tensor: torch.Tensor) -> Image:
    # We assume tensor has:
    # 1. values in [-1, 1]
    # 2. shape (B, C, H, W)
    B, C, H, W = tensor.shape

    # Finding the greatest divisors of B
    rows, cols = 1, B
    for i in reversed(range(1, int(sqrt(B)))):
        if B % i == 0:
            rows, cols = i, B // i
            break
    tensor = tensor.view(rows, cols, C, H, W)

    irows, icols, iC, iH, iW = 0, 1, 2, 3, 4
    tensor = tensor.permute(iC, irows, iH, icols, iW).reshape(C, rows * H, cols * W)

    tensor = ((tensor + 1.0) / 2.0).clamp(0, 1)
    return tvtf.functional.to_pil_image(tensor)


def runway_stable_diffusion_v1_5():
    source = "runwayml/stable-diffusion-v1-5"
    unet = UNet2DConditionModel.from_pretrained(
        source, use_safetensors=True, subfolder="unet"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        source, use_safetensors=True, subfolder="vae"
    ).to(device)
    # feature_extractor = CLIPImageProcessor.from_pretrained(source, use_safetensors=True).to(device)
    scheduler = PNDMScheduler.from_pretrained(source, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        source, use_safetensors=True, subfolder="text_encoder"
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer")

    prompts = ["a corgi", "brad pitt"]
    prompt_balancing_weight = 0.55
    seed = 2
    height, width = vae.config.sample_size, vae.config.sample_size
    num_inference_steps = 100
    guidance_scale = 10
    start_from_image = None  # "alpine_landscape.jpg"
    n_initial_steps_to_skip = 20 if start_from_image is not None else 0
    lag = 0  # np.round(num_inference_steps * 0.2).astype(np.int32)

    generator = torch.manual_seed(seed)
    tokenizer_kwargs = dict(
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input = tokenizer(prompts, **tokenizer_kwargs)
    prompts = [0]  # hack to get back to len(prompts) == 1
    batch_size = len(prompts)
    no_text_input = tokenizer([""] * len(prompts), **tokenizer_kwargs)
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_embeddings = text_encoder(no_text_input.input_ids.to(device))[0]
    text_embeddings = (
        prompt_balancing_weight * text_embeddings[0]
        + (1 - prompt_balancing_weight) * text_embeddings[1]
    ).unsqueeze(0)

    text_embeddings = torch.cat(
        [uncond_embeddings, text_embeddings], dim=0
    )  # concat on batch dim
    B, T, D = text_embeddings.shape  # batch, sequence_len, dictionary_length

    scale_factor = 2 ** (len(vae.config.down_block_types) - 1)

    scheduler.set_timesteps(num_inference_steps)

    # Initializing the first latent.
    # Method 1: from Gaussian noise.
    initial_noisy_latents = torch.randn(
        (
            batch_size,
            unet.config.in_channels,
            height // scale_factor,
            width // scale_factor,
        ),
        generator=generator,
    ).to(device)
    initial_noisy_latents *= scheduler.init_noise_sigma

    # Method 2: from an existing image.
    if start_from_image is not None:
        source_img = (
            (
                tvtf.ToTensor()(Image.open(OUTPUT_IMAGES_PATH / start_from_image)) * 2.0
                - 1.0
            )
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            source_latent = (
                vae.encode(source_img).latent_dist.sample() * vae.config.scaling_factor
            )
        initial_noisy_latents = (
            scheduler.add_noise(
                source_latent,
                torch.randn_like(source_latent),
                scheduler.timesteps[n_initial_steps_to_skip],
            )
            .to(device)
            .float()
        )

    latents = initial_noisy_latents
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        if i < n_initial_steps_to_skip:
            continue

        unet_input = torch.cat(
            [latents] * 2, dim=0
        )  # because batch size of text_embeddings is 2.
        unet_input = scheduler.scale_model_input(unet_input, timestep=t)

        with torch.no_grad():
            estimated_noise = unet(
                unet_input,
                scheduler.timesteps[max(0, i - lag)],
                encoder_hidden_states=text_embeddings,
            ).sample

        uncond_noise, cond_noise = estimated_noise.chunk(2)
        estimated_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
        latents = scheduler.step(estimated_noise, t, latents).prev_sample

        # Is (apparently) equivalent to:
        # denoised_latents = scheduler.step(estimated_noise, t, latents).prev_sample

        # uncond_latents, cond_latents = denoised_latents.chunk(2)
        # latents = uncond_latents + guidance_scale * (cond_latents - uncond_latents)

    latents /= vae.config.scaling_factor
    with torch.no_grad():
        img = vae.decode(latents).sample

    B, C, H, W = 0, 1, 2, 3
    img = Image.fromarray(
        ((img / 2.0 + 0.5).clamp(0, 1) * 255)
        .round()
        .permute(B, H, W, C)[0]
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    img.save((OUTPUT_IMAGES_PATH / source.replace("/", "_")).with_suffix(".jpg"))


def custom_google_ddpm_cat_256():
    timesteps = 100

    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
    model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
        device
    )
    scheduler.set_timesteps(timesteps)

    sample_size = model.config.sample_size
    noise_image = torch.randn((1, 3, sample_size, sample_size)).to(device)

    last_generated_image = noise_image
    for timestep in tqdm(scheduler.timesteps):
        with torch.no_grad():
            predicted_noise = model(last_generated_image, timestep).sample
        # last_generated_image -= predicted_noise
        last_generated_image = scheduler.step(
            predicted_noise, timestep, last_generated_image
        ).prev_sample

    generated_image = last_generated_image
    B, C, H, W = 0, 1, 2, 3
    generated_image = Image.fromarray(
        ((generated_image / 2 + 0.5).clamp(0, 1).permute(B, H, W, C)[0] * 255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    generated_image.save(OUTPUT_IMAGES_PATH / "custom.jpg")


def google_ddpm_cat_256():
    ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
        device
    )

    image = ddpm(num_inference_steps=100).images[0]
    image.save(OUTPUT_IMAGES_PATH / "google_cat.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--notes", type=str, help="Notes about the experiment")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-s", "--skip-training", action="store_true")

    args = parser.parse_args()

    custom_ddpm(args.notes, args.debug, args.skip_training)
