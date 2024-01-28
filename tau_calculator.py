#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:03:20 2023

@author: marcorax93
"""

import numpy as np

num_hidden = 10
camp_period = 0.0005
betas = np.linspace(0.4,0.8,num_hidden)
taus =  -camp_period/np.log(betas)
freqs =  1/taus