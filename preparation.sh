conda create -n omnibind python=3.8
conda activate omnibind
# replace the package manager if needed
apt install libgeos++-dev
# install imagebind
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install .
# install omnibind envs, overwrite imagebind envs
cd ..
pip install -r requirements.txt
