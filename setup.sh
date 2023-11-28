#!/bin/bash

apt-get update
apt-get install -y python3-pip git neovim ffmpeg
pip3 install numpy pandas matplotlib ipython ipdb requests tqdm wandb
pip3 install diffusers accelerate transformers datasets bitsandbytes evaluate scikit-learn
pip3 install torch torchvision torchaudio tensorboard --index-url https://download.pytorch.org/whl/cu118

