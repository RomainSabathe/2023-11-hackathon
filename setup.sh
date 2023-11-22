#!/bin/bash

apt-get update
apt-get install -y python3-pip git neovim
pip3 install numpy pandas matplotlib ipython ipdb tensorboard requests
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

