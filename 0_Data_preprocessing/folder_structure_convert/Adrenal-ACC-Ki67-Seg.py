import os
from glob import glob
import os.path as osp
import shutil

caseDir = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg/derived/nifti'
rootP = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg'
os.makedirs(osp.join(rootP, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(rootP, 'labelsTr'), exist_ok = True)

for case in os.listdir(caseDir):
    img = sorted(glob(osp.join(caseDir, case, 'img*')))[0]
    label = glob(osp.join(caseDir, case, 'seg*'))[0]
    
    shutil.move(img, osp.join(rootP, 'imagesTr', case + '.nii.gz'))
    shutil.move(label, osp.join(rootP, 'labelsTr', case + '.nii.gz'))