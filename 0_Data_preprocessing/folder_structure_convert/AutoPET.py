import os
from glob import glob
import os.path as osp
import shutil

caseDir = '/home/nute11a/nfs_server/dataset/original/multi-modality/AutoPET/nii_images'
rootP = '/home/nute11a/nfs_server/dataset/original/multi-modality/AutoPET'
saveP_PET = '/home/nute11a/nfs_server/dataset/original/PET/AutoPET(PET)'
saveP_CT = '/home/nute11a/nfs_server/dataset/original/CT/AutoPET(CT)'
os.makedirs(osp.join(saveP_PET, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(saveP_PET, 'labelsTr'), exist_ok = True)
os.makedirs(osp.join(saveP_CT, 'imagesTr'), exist_ok = True)
os.makedirs(osp.join(saveP_CT, 'labelsTr'), exist_ok = True)

for case in os.listdir(caseDir):
    
    try:
        img_CT = sorted(glob(osp.join(caseDir, case, '*', 'CTres.nii.gz')))[0]
        img_PET = sorted(glob(osp.join(caseDir, case, '*', 'PET.nii.gz')))[0]
        label = glob(osp.join(caseDir, case, '*', 'SEG.nii.gz'))[0]
        shutil.move(img_CT, osp.join(saveP_CT, 'imagesTr', case + '.nii.gz'))
        shutil.move(img_PET, osp.join(saveP_PET, 'imagesTr', case + '.nii.gz'))
        shutil.copy(label, osp.join(saveP_CT, 'labelsTr', case + '.nii.gz'))
        shutil.move(label, osp.join(saveP_PET, 'labelsTr', case + '.nii.gz'))
    except:
        print()