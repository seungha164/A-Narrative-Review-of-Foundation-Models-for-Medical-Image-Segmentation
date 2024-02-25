import os
from glob import glob
import os.path as osp
import shutil

caseDir = '/home/nute11a/nfs_server/dataset/original/CT/Adrenal-ACC-Ki67-Seg/derived/nifti'
rootP = '/home/nute11a/nfs_server/dataset/original/CT/PleThora'
os.makedirs(osp(rootP, 'imagesTr'), exist_ok = True)
os.makedirs(osp(rootP, 'labelsTr'), exist_ok = True)

for case in os.listdir(caseDir):
    img = sorted(glob(osp(caseDir, case, 'img*')))[0]
    label = glob(osp(caseDir, case, 'seg*'))
    
    shutil.move(img, osp(rootP, 'imagesTr', case + '.nii.gz'))
    shutil.move(label, osp(rootP, 'labelsTr', case + '.nii.gz'))