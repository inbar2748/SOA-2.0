import math
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL.Image import Resampling
from scipy.stats import binom
from skimage.filters import frangi
from skimage.morphology import remove_small_objects
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw  # Import ImageDraw from PIL
from skimage import morphology, measure


def binomial_distribution(num_lines):
    # setting the values of n and p
    # defining the list of k values
    n = num_lines
    p = 1 / n
    s_values = list(range(n))
    # list of pmf values
    gbcountRANDOM = []
    for k in s_values:
        ans = (binom.pmf(k, n, p)) * n
        if float(ans) > 0.05:
            ans_ceil = int(math.ceil(ans))
            gbcountRANDOM.append(ans_ceil)
    # printing the table
    print('\n', "<--------------- Binomial distribution simulation: --------------->", '\n')
    #print("Random simulation Classification")
    print(gbcountRANDOM,'\n')

    return gbcountRANDOM

