
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

def normalizePlanes(npzarray):
        """
        Normalizing the image using the appropriate maximum and minimum values associated 
        with a CT scan for lung cancer (in terms of Hounsfeld Units)
        
        """
        max_hu = np.max(npzarray)
        min_hu= np.min(npzarray)#-1000.
        npzarray = (npzarray - min_hu) / (max_hu - min_hu)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray