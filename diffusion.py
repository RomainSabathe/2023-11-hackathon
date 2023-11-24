from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
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


def custom_ddpm():
    T = 1_000  # like in the paper.
    beta_1, beta_T = 1e-4, 0.02
    dataset_root = Path("data") / "fashion_mnist"
    dataset = FashionMNIST(
        dataset_root,
        download=not dataset_root.exists(),
        transform=tvtf.Compose(
            [
                tvtf.ToImage(),
                tvtf.ToDtype(torch.float32, scale=True),
                tvtf.Resize((32, 32), antialias=True),
                # lambda x: (x * 2.0) - 1.0,
                lambda x: (x * 1.0) - 0.5,
            ]
        ),
    )

    class Subset:
        def __init__(self, dataset, n):
            self.dataset = dataset
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.dataset[i]

    # dataset = Subset(dataset, 4)

    scale_factor = 10
    batch_size = 64 * scale_factor
    batch_size = 128

    # batch_size = 4

    learning_rate = 4e-3
    n_epochs = 5
    n_iter_per_epoch = 9999999
    seed = 1
    skip_training = True

    torch.manual_seed(seed)

    sample_image, sample_label = dataset[0]
    C, H, W = sample_image.shape
    model = UNet2DModel(
        sample_size=(H, W),
        in_channels=C,
        out_channels=C,
        block_out_channels=(32, 64, 128, 128),
    ).to(device)

    with torch.no_grad():
        beta = torch.linspace(beta_1, beta_T, steps=T).to(device)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

    if not skip_training:

        def q_given_x0(ts, x0s) -> torch.Tensor:
            # ts has shape (B,) and dtype torch.int
            # x0s has shape (B, C, H, W) and dtype torch.float
            B, C, H, W = x0s.shape
            _alpha_bar = alpha_bar[ts].view(B, 1, 1, 1)
            return torch.normal(
                mean=torch.sqrt(_alpha_bar) * x0s,
                std=torch.sqrt(1 - _alpha_bar),
            )

        # Not used in practice since q_given_x0 is a generalisation of this formula.
        def q_given_xtm1(t, xtm1) -> torch.Tensor:
            return torch.normal(mean=torch.sqrt(1 - beta(t)) * xtm1, std=beta(t))

        # Checking it all works
        with torch.no_grad():
            ts = torch.tensor([0, 10, 100, 500, 800, 999]).to(device)
            noisy_sample_images = q_given_x0(
                ts, torch.stack([sample_image] * len(ts), dim=0).to(device)
            )
            # noisy_sample_images = ((noisy_sample_images + 1.0) / 2.0).clamp(0, 1)
            noisy_sample_images = ((noisy_sample_images - 0.5) / 1.0).clamp(-0.5, 0.5)
            for t, noisy_sample_image in zip(ts, noisy_sample_images):
                tvtf.ToPILImage()(noisy_sample_image).save(
                    OUTPUT_IMAGES_PATH / f"forward_pass_{t:04d}.jpg"
                )

        # reminder:
        # forward process goes from image to noise
        #                             0   to  T
        # backward process goes from noise to image
        #                             T   to  0
        # algo:
        # 1. get a batch of images Xs and
        #    sample different timesteps Ts
        # 2. for each X, compute the forward process with the respective T and T+1
        # (we will train the model to perform the backward process from T+1 to T)
        # forward(x, t) = q_given_x0(t, x)
        # forward(x, t+1) = q_given_x0(t+1, x)
        # 3. Determine the noise pattern that was added during the forward step from T to T+1:
        # noise = forward(x, t+1) - forward(x, t)
        # 4. Get the prediction of the UNet from the image at step t+1
        # predicted_noise = UNet(x(t+1)) = UNet(forward(x, t+1))
        # 5. Minimise the MSE
        # grad((noise - predicted_noise)**2.)

        # Training loop
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=n_epochs * len(dataset)
        )
        for epoch in tqdm(range(n_epochs)):
            for i, (x0s, _) in enumerate(
                tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True))
            ):
                if i >= n_iter_per_epoch:
                    break

                B, C, H, W = x0s.shape
                # with torch.no_grad():
                ts = torch.randint(
                    low=0, high=T - 1, size=(B,), dtype=torch.long, device=device
                )
                x0s = x0s.to(device)

                noises = torch.randn_like(x0s, device=device)
                xtp1s = (
                    torch.sqrt(alpha_bar[ts]).view(B, 1, 1, 1) * x0s
                    + torch.sqrt(1 - alpha_bar[ts]).view(B, 1, 1, 1) * noises
                )

                # xts = q_given_x0(ts, x0s)
                # xtp1s = q_given_x0(ts + 1, x0s)
                # noises = xtp1s - xts
                predicted_noises = model(xtp1s, ts).sample
                loss = loss_fn(predicted_noises, noises)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if i % 25 == 0:
                    print(loss)

        torch.save(model.cpu().state_dict(), "/tmp/unet.bin")

    # Reverse process now.
    # recall: p(x_t-1|x_t) ~ N(mu_t, sigma_t) with:
    # sigma**2_t = beta_t or beta_tilde_t
    # mu_t = complicated formula
    model.load_state_dict(torch.load("/tmp/unet.bin"))
    model = model.to(device).eval()
    batch_size = 4

    @torch.no_grad()
    def sample(model, sz=(1, 1, 32, 32)):
        n_steps = 1_000
        βmin = 1e-4
        βmax = 0.02

        device = next(model.parameters()).device
        x_t = torch.randn(sz, device=device)
        β = torch.linspace(βmin, βmax, n_steps)
        α = 1.0 - β
        ᾱ = torch.cumprod(α, dim=0)
        σ = β.sqrt()

        for t in tqdm(reversed(range(n_steps))):
            t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
            z = (torch.randn(x_t.shape) if t > 0 else torch.zeros(x_t.shape)).to(device)
            ᾱ_t1 = ᾱ[t - 1] if t > 0 else torch.tensor(1)
            b̄_t = 1 - ᾱ[t]
            b̄_t1 = 1 - ᾱ_t1
            noise_pred = model(x_t, t_batch).sample
            # x_0_hat = ((x_t - b̄_t.sqrt() * noise_pred) / ᾱ[t].sqrt()).clamp(-1, 1)
            x_0_hat = ((x_t - b̄_t.sqrt() * noise_pred) / ᾱ[t].sqrt()).clamp(-0.5, 0.5)
            x0_coeff = ᾱ_t1.sqrt() * (1 - α[t]) / b̄_t
            xt_coeff = α[t].sqrt() * b̄_t1 / b̄_t
            x_t = x_0_hat * x0_coeff + x_t * xt_coeff + σ[t] * z

            if t % 100 == 0 or (t < 100 and t % 10 == 0):
                img = x_t[0]
                # img = ((img + 1.0) / 2.0).clamp(0, 1)
                img = ((img + 0.5) / 1.0).clamp(-0.5, 0.5)
                tvtf.ToPILImage()(img).save(
                    OUTPUT_IMAGES_PATH / f"reverse_pass_{t:04d}.jpg"
                )
        return x_t

    la = sample(model)
    return

    with torch.no_grad():
        xts = torch.randn((batch_size, C, H, W)).to(device)
        for t in tqdm(reversed(range(T))):
            ts = t * torch.ones((batch_size, 1, 1, 1), device=device, dtype=torch.long)
            z = (
                torch.randn_like(xts, device=device)
                if t > 0
                else torch.zeros_like(xts, device=device)
            )

            std = torch.sqrt(beta[ts])

            estimated_noise = model(xts, ts.view(batch_size)).sample
            mu = alpha[ts] ** (-1 / 2) * (
                xts - estimated_noise * (beta[ts] / torch.sqrt(1 - alpha_bar[ts]))
            )

            xts = (mu + std * z).clamp(-1, 1)

            if t % 100 == 0 or (t < 100 and t % 10 == 0):
                img = xts[0]
                img = ((img + 1.0) / 2.0).clamp(0, 1)
                tvtf.ToPILImage()(img).save(
                    OUTPUT_IMAGES_PATH / f"reverse_pass_{t:04d}.jpg"
                )

    import ipdb

    ipdb.set_trace()


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
    custom_ddpm()
