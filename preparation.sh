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
# install KNN_CUDA pytorch
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
make && make install
cd ..
# install pointnet2 ops
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# if use cuda12, use command below instead
git clone https://github.com/HeBangYan/pointnet2_ops_CUDA12.0.git
cd pointnet2_ops_CUDA12.0
pip install pointnet2_ops_lib/.