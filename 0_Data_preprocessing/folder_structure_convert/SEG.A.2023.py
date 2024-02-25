import os
from glob import glob
import os.path as osp
import shutil
from tqdm import tqdm
from utils.format_convert import nrrd2nii
#caseDir = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg/derived/nifti'
rootP = '/home/nute11a/nfs_client3090/dataset/original/CT/SEG.A.2023'
saveP = '/home/nute11a/workspace/SAMM/result/SEG.A.2023'
os.makedirs(osp.join(saveP, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(saveP, 'labelsTr'), exist_ok = True)

for nrrd_data in tqdm(glob(osp.join(rootP, "*", "*", "*.nrrd"))):

    if '.seg.nrrd' in nrrd_data.split('/')[-1]:
        nifti_name = nrrd_data.split('/')[-1].replace('.seg.nrrd', '.nii.gz')
        #if osp.isfile(osp.join(saveP, 'imagesTr', nifti_name)): continue
        nrrd2nii(nrrd_data, osp.join(saveP, 'imagesTr', nifti_name))
    else:
        nifti_name = nrrd_data.split('/')[-1].replace('.nrrd', '.nii.gz')
        #if osp.isfile(osp.join(saveP, 'labelsTr', nifti_name)): continue
        nrrd2nii(nrrd_data, osp.join(saveP, 'labelsTr', nifti_name))
