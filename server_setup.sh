# Update
sudo apt update

# Install pyenv dependencies
sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y

# Upgrade other system packages
sudo apt upgrade -y

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv init to .profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> .profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> .profile
echo 'eval "$(pyenv init -)"' >> .profile

# Reload shell
exec $SHELL -l

pyenv install 3.10.2

# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -

# Configure poetry
source $HOME/.poetry/env

# Clone project
git clone https://github.com/datapointchris/object_detection_keras_unet.git

# Move into directory
cd object_detection_keras_unet/

# Install project
poetry install