#!/bin/bash

echo "Installing dependencies for macOS ARM64 (M1/M2/M3)..."

# Install conda packages first
echo "Installing conda packages..."
conda install -y -c conda-forge pip wheel cython

# Install gevent from conda-forge (binary distribution)
echo "Installing gevent from conda-forge..."
conda install -y -c conda-forge "gevent=22.10.2"

# Install gevent-websocket and other dependencies
echo "Installing gevent-websocket and core dependencies..."
pip install gevent-websocket==0.10.1 pyzmq==25.1.1 socketio-client==0.7.2 runstats

# Install openfabric-pysdk with specific version
echo "Installing openfabric-pysdk..."
pip install openfabric-pysdk==0.2.9

# Create a temporary requirements file without the packages we've already installed
echo "Installing remaining packages..."
grep -v -E "gevent|gevent-websocket|pyzmq|socketio-client|runstats|openfabric-pysdk" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt
rm requirements_temp.txt

echo "Installation complete!"
