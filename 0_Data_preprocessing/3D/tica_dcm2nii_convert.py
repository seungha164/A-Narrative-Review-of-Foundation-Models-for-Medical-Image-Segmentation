import dicom2nifti
import os
from glob import glob
from tqdm import tqdm
import pydicom
import numpy as np

#* (1) PleThora
ds = 'PleThora'
dcmP = '/home/nute11a/dataset/CT/original/NSCLC Pleural Effusion/manifest-1586193031612/NSCLC-Radiomics'
outP = '/home/nute11a/dataset/CT/original/NSCLC Pleural Effusion/images_nii'
#* (2) HCC-TACE-Seg
ds = 'HCC-TACE-Seg'
dcmP = '/home/nute11a/dataset/CT/original/HCC-TACE-Seg/manifest-1643035385102/HCC-TACE-Seg'
outP = '/home/nute11a/dataset/CT/original/HCC-TACE-Seg/images_nii'

if ds == 'PleThora':
    cases = glob(f'{dcmP}/*/*/*')
    for case in tqdm(cases):
        patient = case.split('/')[-3]
        os.makedirs(f'{outP}/{patient}',exist_ok=True)
        dicom2nifti.convert_directory(case, f'{outP}/{patient}')
elif ds == 'HCC-TACE-Seg':
    cases = glob(f'{dcmP}/*/*/*')
    for case in tqdm(cases):
        patient = case.split('/')[-3]
        os.makedirs(f'{outP}/{patient}/{case.split("/")[-1]}',exist_ok=True)
        if len(glob(f'{case}/*')) == 1: #! segmentatino
            dcm = pydicom.dcmread(glob(f'{case}/*')[0])
            arr = dcm.pixel_array
            np.save(f'{outP}/{patient}/{case.split("/")[-1]}/segmentation.npy', arr)
        else:
            continue
            dicom2nifti.convert_directory(case, f'{outP}/{patient}/{case.split("/")[-1]}')
