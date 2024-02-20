
import numpy as np

def checkRatio(arr, h, w):
    ratio = (np.count_nonzero(arr) / (h * w)) * 100
    if ratio < 0.153:
        return False
    else:
        return True
    
def exploit_mask(mask, pixel):
    m = mask.copy()
    m[m != pixel] = 0
    m[m == pixel] = 255
    return m