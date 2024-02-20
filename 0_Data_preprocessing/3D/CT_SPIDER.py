from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json
import SimpleITK as sitk
import custom_utils as cutils
import read_utils as rutils


def getImageFiles(segfiles, ds):
    if ('MSD-' in ds) or (ds == 'PASeg') or (ds == 'SPIDER'):
        return [seg.replace('labelsTr', 'imagesTr') for seg in segfiles]
    if ds == 'ATLAS2023':
        return [seg.replace('labelsTr', 'imagesTr').replace('Tr/lb', 'Tr/im') for seg in segfiles]
    if ds == 'TDSC-ABUS2023':
        return [seg.replace('labelsTr', 'imagesTr').replace('MASK', 'DATA') for seg in segfiles]
    if ds == 'RESECT(US)':
        return [seg.replace('labelsTr', 'imagesTr').replace('-after-resection', '-after').replace('-before-tumor', '-before').replace('-during-resection', '-during') for seg in segfiles]
    if ds == 'LiTS17':
        return [seg.replace('labelsTr', 'imagesTr').replace('segmentation', 'volume') for seg in segfiles]
    if ds == '3DLSC-COVID':
        return [seg.replace('labelsTr', 'imagesTr').replace('.nii', '.nii.gz') for seg in segfiles]
    if ds == 'SegRap2023-Task001' or ds == 'SegRap2023-Task002':
        imgs, task = [], ds.split('-')[-1]
        for seg in segfiles:
            case = seg.split('/')[-1].replace('.nii.gz', '')
            imgs.append(seg.replace('SegRap2023_Training_Set_120cases_OneHot_Labels', 'SegRap2023_Training_Set_120cases').replace(case, 'image_contrast').replace(task, case))
        return imgs
    
def saveMask(mask, saveMaskRoot, saveMaskPatient, z_axis, h, w, clsNames):
    is_save_img = False
    #! Multi-class 처리
    # 1. class 추출
    classes = np.unique(mask).tolist()
    classes.remove(0)           # 0: background
    # 2. 각 class별 mask 저장
    for cls in classes:
        mask_cls = cutils.exploit_mask(mask, cls) # 해당 cls의 label만 추출
        #! 동일 class multi-object 처리
        eq_cls_cnt = 0
        mask_cls = mask_cls.astype(np.uint8)
        contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for con in contours:
            mask_zeros = np.zeros([h, w, 1], dtype=np.uint8)
            cv2.fillPoly(mask_zeros, [con], 255)
            if not cutils.checkRatio(mask_zeros, h, w): continue
            mask_zeros = np.repeat(mask_zeros, 3, axis = 2)
            mask_zeros = mask_zeros.astype(np.float32)
            if type(clsNames) == 'dict':
                cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}/{z_axis}_{int(cls)}_{clsNames[int(cls)]}_{eq_cls_cnt}.png', mask_zeros)
            else:
                cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}/{z_axis}_{int(cls)}_{clsNames[int(cls)]}_{eq_cls_cnt}.png', mask_zeros)
            eq_cls_cnt += 1                # 동일 class의 다른 객체들 처리해주기 위한 idx
            is_save_img = True      # mask 저장시 -> 무조건 img도 저장
    return is_save_img


def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)
    
    #* (2) images/labels 얻기
    filesSeg = glob(cfg["segP"])
    filesImg = getImageFiles(filesSeg, cfg["dataset_name"])

    #* (3) 3D -> 2D convert/save
    for idx in tqdm(range(len(filesImg))):
        imgName, segName = filesImg[idx], filesSeg[idx]
        patientName = segName.split('/')[-1].replace(cfg["format"], "")
        os.makedirs(f"{cfg['save_imgP']}/{patientName}", exist_ok=True)
        os.makedirs(f"{cfg['save_segP']}/{patientName}", exist_ok=True)
        #* 파일 읽기
        segdata, imgdata = sitk.ReadImage(segName), sitk.ReadImage(segName)
        segdata, imgdata = sitk.GetArrayViewFromImage(segdata), sitk.GetArrayViewFromImage(imgdata)
        #_, segdata = rutils.read_by_format(segName, cfg["format"])
        #_, imgdata = rutils.read_by_format(imgName, cfg["format"])
        
        if cfg["dataset_name"] == "PASeg":
            imgdata = imgdata[ : , : , : , 0]   # [x, y, z, 1] -> [x, y, z]
        # check
        if imgdata.shape != segdata.shape:
            print('NOT SUIT : ', segName)
            break
        if cfg["format"] in ['.nii.gz', '.nrrd', '.nii']:
            h, w, z = segdata.shape
        elif cfg["format"] in ['.mha', '.mhd']:
            z, h, w = segdata.shape
        #print(filesSeg[idx], ' > ', np.unique(segdata))
        #continue
        #! 2D img로 보기 - z divide
        for _z in range(z):
            if cfg["format"] in ['.nii.gz', '.nrrd', '.nii']:
                mask, img = segdata[:,:,_z], imgdata[:,:,_z]
            else:
                mask, img = segdata[_z], imgdata[_z]
            if not cutils.checkRatio(mask, h, w):   continue
            #* mask 처리
            is_save_img = saveMask(mask, cfg["save_segP"], patientName, _z, h, w, cfg["CLASSES_NAME"])
            #* image 처리
            if is_save_img:
                cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}/{_z}.png', img)

if __name__ == "__main__":
    main(
        config_file = './configs/SPIDER.json'
    )