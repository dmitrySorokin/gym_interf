#!/usr/bin/env bash

cd gym_interf/envs/cpp/

# clean-up cpp
rm -rf build/*

# compile & install cpp
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../libs/ ../
make install -j7

# install pip package
cd ../../../../
pip3 install -e ./