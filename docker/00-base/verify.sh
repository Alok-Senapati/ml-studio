#!/usr/bin/env bash

set -e

echo "=============================="
echo "AI Workstation Base Verification"
echo "=============================="

echo
echo "Python"
python --version

echo
echo "Pip"
pip --version

echo
echo "Git"
git --version

echo
echo "Workspace"
pwd

echo
echo "Operating System"
cat /etc/os-release

echo
echo "CUDA Runtime"
echo $CUDA_VERSION

echo
echo "PATH"
echo $PATH

echo
echo "Verification Completed Successfully"