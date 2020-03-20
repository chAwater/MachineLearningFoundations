#!/bin/bash
###############################################################################
# 
# Author: chenhe
# Created Time: 2020.03.20 13:28:53
# 
###############################################################################


sed -E 's@<img src="http://latex.codecogs.com/svg.latex\?(.+)"/>@<img src="http://latex.codecogs.com/svg.latex?{\1}"/>@' README.md
