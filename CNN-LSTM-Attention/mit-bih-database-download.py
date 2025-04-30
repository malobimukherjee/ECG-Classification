import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

if os.path.isdir("mitdb"):
 print('You already have the data.')
else:
 wfdb.dl_database('mitdb', 'mitdb')