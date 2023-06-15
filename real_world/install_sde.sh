set -e
mkdir -p tmp
cd tmp

# Replace outdated GPG key
# Ref: https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo rm -f /etc/apt/sources.list.d/cuda.list
sudo rm -f /etc/apt/sources.list.d/nvidia-ml.list

#update apt repos
sudo apt update
#sudo apt upgrade -y
#sudo apt upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
# sudo apt install -y vim
sudo apt install -y tmux 
# sudo apt install -y git
sudo apt install -y python3-distutils
sudo apt install -y python3-pip
pip3 install --upgrade pip

curl -sSL "http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run" -o cuda.run
chmod +x cuda.run
curl -sSL "http://developer.download.nvidia.com/compute/redist/cudnn/v8.0.4/cudnn-11.0-linux-x64-v8.0.4.30.tgz" -o cudnn.tgz
sudo ./cuda.run --silent --toolkit --override
sudo tar --no-same-owner -xzf cudnn.tgz -C /usr/local
cd ..
# pip3 install -r ~/score-based-uda-digit/requirements.txt
pip3 install -r ~/csm/requirements.txt
pip3 install --upgrade "jax[cuda110]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Note: requires running the following before using `python` command:
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
