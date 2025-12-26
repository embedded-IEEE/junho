#!/bin/sh

set -e

password=$1

# install SCSservo pyserial imutils
apt update
python3 setup.py install
