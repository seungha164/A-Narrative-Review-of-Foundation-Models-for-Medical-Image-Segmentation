import os
from glob import glob
import os.path as osp
import shutil
from tqdm import tqdm
from utils.format_convert import nrrd2nii
#caseDir = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg/derived/nifti'
rootP = '/home/nute11a/nfs_client3090/dataset/original/Ultrasound/TDSC-ABUS2023'
saveP = '/home/nute11a/workspace/SAMM/result/TDSC-ABUS2023'
os.makedirs(osp.join(saveP, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(saveP, 'labelsTr'), exist_ok = True)

for nrrd_data in tqdm(glob(osp.join(rootP, 'imagesTr', "*"))):
    nifti_name = nrrd_data.split('/')[-1].replace('.nrrd', '.nii.gz').split('_')[-1]
    nrrd2nii(nrrd_data, osp.join(saveP, 'imagesTr', nifti_name))

for nrrd_data in tqdm(glob(osp.join(rootP, 'labelsTr', "*"))):
    nifti_name = nrrd_data.split('/')[-1].replace('.nrrd', '.nii.gz').split('_')[-1]
    nrrd2nii(nrrd_data, osp.join(saveP, 'labelsTr', nifti_name))
