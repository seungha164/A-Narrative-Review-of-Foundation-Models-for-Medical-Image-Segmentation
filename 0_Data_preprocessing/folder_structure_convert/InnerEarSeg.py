import os
from glob import glob
import os.path as osp
import shutil
from tqdm import tqdm
from utils.format_convert import nii2nii
#caseDir = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg/derived/nifti'
rootP = '/home/nute11a/nfs_client3090/dataset/original/CT/InnerEarSeg/data/using_data'
saveP = '/home/nute11a/workspace/SAMM/result/InnerEarSeg'
os.makedirs(osp.join(saveP, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(saveP, 'labelsTr'), exist_ok = True)

for nrrd_data in tqdm(glob(osp.join(rootP, 'Training CT-scans', "*"))):
    nifti_name = nrrd_data.split('/')[-1].replace('.nii', '.nii.gz')
    nii2nii(nrrd_data, osp.join(saveP, 'imagesTr', nifti_name))

for nrrd_data in tqdm(glob(osp.join(rootP, 'Validation CT-scans', "*"))):
    nifti_name = nrrd_data.split('/')[-1].replace('.nii', '.nii.gz')
    nii2nii(nrrd_data, osp.join(saveP, 'imagesTr', nifti_name))
    
for nrrd_data in tqdm(glob(osp.join(rootP, 'Training Manual segmentation', "*"))):
    nifti_name = nrrd_data.split('/')[-1].replace('.nii', '.nii.gz')
    nii2nii(nrrd_data, osp.join(saveP, 'labelsTr', nifti_name))

for nrrd_data in tqdm(glob(osp.join(rootP, 'Validation Manual segmentation', "*"))):
    nifti_name = nrrd_data.split('/')[-1].replace('.nii', '.nii.gz')
    nii2nii(nrrd_data, osp.join(saveP, 'labelsTr', nifti_name))
