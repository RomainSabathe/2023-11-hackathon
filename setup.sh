#!/bin/bash

apt-get update
apt-get install -y python3-pip git neovim
pip3 install numpy pandas matplotlib ipython ipdb requests tqdm
pip3 install diffusers accelerate transformers
pip3 install torch torchvision torchaudio tensorboard --index-url https://download.pytorch.org/whl/cu118

