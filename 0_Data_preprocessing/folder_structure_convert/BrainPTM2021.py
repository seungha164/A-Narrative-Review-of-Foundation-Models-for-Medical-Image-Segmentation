import os
from glob import glob
import os.path as osp
import shutil
from tqdm import tqdm
from utils.format_convert import mhd2nii
#caseDir = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg/derived/nifti'
rootP = '/home/nute11a/nfs_client3090/dataset/original/MR/BrainPTM2021/original_data'
saveP = '/home/nute11a/workspace/SAMM/result/BrainPTM2021'
os.makedirs(osp.join(saveP, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(saveP, 'labelsTr'), exist_ok = True)

#for mha_data in tqdm(glob(osp.join(rootP, 'imagesTr', "*"))):
#    nifti_name = mha_data.split('/')[-1].replace('.mha', '.nii.gz')
#    mhd2nii(mha_data, osp.join(saveP, 'imagesTr', nifti_name))
for case in tqdm(glob(osp.join(rootP, "*"))):
    T1_img = osp.join(rootP, "*")
    
for mha_data in tqdm(glob(osp.join(rootP, 'labelsTr', "*"))):
    try:
        nifti_name = mha_data.split('/')[-1].replace('.mha', '.nii.gz')
        mhd2nii(mha_data, osp.join(saveP, 'labelsTr', nifti_name)) 
    except:
        print(mha_data)