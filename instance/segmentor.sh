#!/bin/bash

PLY_PATH=$1
relu=$2
num=$3
cd ./instance/Segmentator/build
./segmentator $PLY_PATH $relu $num