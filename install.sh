#!/usr/bin/env bash

cd gym_interf/envs/cpp/
pwd

# clean-up cpp
rm -rf build

# compile & install cpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../libs/ ../
make install -j7

# install pip package
cd ../../../../
pip3 install -e ./
