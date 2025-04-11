set -E
sudo apt update
sudo apt upgrade
sudo apt install python3-pip build-essential libevent-dev automake autoconf libtool pkg-config autoconf-archive
pip install python-binary-memcached
./autogen.sh
./configure
