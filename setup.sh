set -E
sudo apt update
sudo apt upgrade
sudo apt install build-essential libevent-dev automake autoconf libtool pkg-config autoconf-archive
./autogen.sh
./configure
