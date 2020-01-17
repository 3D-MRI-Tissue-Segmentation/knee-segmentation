# FOR MAC INSTALL ONLY, NOT LINUX OR WINDOWS
# Panda3D is a bit all over the place; it needs a 3rd party boi to compile, 
# which you have to unzip and move into the panda3d source directory, and 
# then needs to be built. 

# This script will download Panda3D as well as the 3rd party build dependen-
# cies and move them into the source directory. 
# The instructions can be found on https://github.com/panda3d/panda3d


pip3 install panda3d   # This will work for all OS


# Download third party packages (to current directory):
wget https://www.panda3d.org/download/panda3d-1.10.5/panda3d-1.10.5-tools-mac.tar.gz 

tar -zxvf panda3d-1.10.5-tools-mac.tar

