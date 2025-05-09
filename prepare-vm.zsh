#/usr/bin/env zsh

# log everything
set -x

# init conda
which conda && conda init zsh
exec zsh

#### Bleder dataset gen

cd

# Download blender
wget https://download.blender.org/release/Blender4.2/blender-4.2.9-linux-x64.tar.xz \
	--directory-prefix="/tmp/"
# Unpack it into local bin
tar xf /tmp/blender-4.2.9-linux-x64.tar.xz --directory "${HOME}/.local/bin"
# Provide path to exec
ln -s "${HOME}/.local/bin/blender-4.2.9-linux-x64/blender" "${HOME}/.local/bin/blender"
# Ensure installation is ok
blender --version

# Download blender dataset script
git clone https://github.com/SherAndrei/blender-gen-dataset.git


#### Ceres Solver with CUDA support

# Building this should resolve COLMAP warning
# ```
# Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support. Falling back to CPU-based sparse solvers.
# ```

# Installation source: http://ceres-solver.org/installation.html

cd

git clone https://ceres-solver.googlesource.com/ceres-solver

sudo apt-get install -y \
    cmake \
    ninja-build \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    cudss

cd ceres-solver
mkdir build
cd builld

cmake .. -DUSE_CUDA=ON -DBUILD_TESTING=OFF -Gninja
ninja
sudo ninja install

#### COLMAP For dataset generation

# Build [COLMAP](https://colmap.github.io/) with GUI and without display to be able to run on server.
# Source: https://github.com/colmap/colmap/issues/1433
# Installation source: https://colmap.github.io/install.html

cd

sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

# Under Ubuntu 22.04, there is a problem when compiling with Ubuntu’s default CUDA package and GCC, and you must compile against GCC 10.
# Source: https://colmap.github.io/install.html
if [ "$NAME" = "Ubuntu" ] && [ "$VERSION_ID" = "22.04" ]; then
	echo "Running on Ubuntu 22.04, compiling COLMAP against GCC 10";
	sleep 2s;
	sudo apt-get install gcc-10 g++-10;
	export CC=/usr/bin/gcc-10;
	export CXX=/usr/bin/g++-10;
	export CUDAHOSTCXX=/usr/bin/g++-10;
else
	echo "Running on ${NAME} ${VERSION_ID}, no additional COLMAP configuration needed";
	sleep 2s;
fi

git clone https://github.com/colmap/colmap.git ~/colmap
cd ~/colmap
mkdir build
cd build

# Specify GPU compute capability of current machine to fix unsupported 'compute_native'
# Source: https://github.com/colmap/colmap/issues/2464
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')

# conda hides libraries, resulting in missing dependencies, e.g. libtiff
# solve it by proceeding with cmake iff environment is default one
# Source: https://github.com/colmap/colmap/issues/188

cmake -DTESTS_ENABLED=OFF -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=${compute_cap} .. -GNinja

ninja
sudo ninja install

cd

#### TensoRF

# Install dependepdencies
conda create -n TensoRF python=3.8 -y
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
pip install plyfile

# Download repo and prepare directories
git clone https://github.com/apchenstu/TensoRF.git ~/TensoRF
mkdir ~/TensoRF/data


