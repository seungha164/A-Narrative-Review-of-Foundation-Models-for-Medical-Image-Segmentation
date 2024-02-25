from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json

import custom_utils as cutils
import read_utils as rutils

def save_Mask(mask, saveMaskRoot, saveMaskPatient, z_axis, h, w, clsNames):
    is_save_img =False
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


def getImageFiles(segfiles, cfg):
    if ('MSD-' in cfg["dataset_name"]) or (cfg["dataset_name"] in ['PASeg', 'SPIDER']):
        return [seg.replace('labelsTr', 'imagesTr') for seg in segfiles]
    if cfg["dataset_name"] == 'InnerEarSeg':
        return [seg.replace('/Training Manual segmentation/', '/Training CT-scans/').replace('/Validation Manual segmentation/', '/Validation CT-scans/') for seg in segfiles]
    if cfg["dataset_name"] == "LiTS17":
        return [seg.replace('labelsTr/segmentation', 'imagesTr/volume') for seg in segfiles]
    if cfg["dataset_name"] == '3DLSC-COVID':
        return [seg.replace('labelsTr', 'imagesTr').replace('.nii', '.nii.gz') for seg in segfiles]
    if cfg["dataset_name"] in ['NSCLC_Pleural_Effusion_1', 'NSCLC_Pleural_Effusion_2']:
        return [glob(f"{cfg['imgP']}/{seg.split('/')[-2]}/*")[0] for seg in segfiles]

def getPatientName(imgName, cfg):
    if cfg['dataset_name'] in ['NSCLC_Pleural_Effusion_1', 'NSCLC_Pleural_Effusion_2']:
        return imgName.split('/')[-2]
    else:
        return imgName.split('/')[-1].replace(cfg["format_img"], '').replace('_', '-')
    
def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)

    #* (2) images/labels 얻기
    filesSeg = glob(cfg["segP"])
    filesImg = getImageFiles(filesSeg, cfg)

    #* (3) save
    for idx in tqdm(range(len(filesImg))):
        imgName, segName = filesImg[idx], filesSeg[idx]
        patientName = getPatientName(imgName, cfg)
        
        os.makedirs(f'{cfg["save_imgP"]}/{patientName}', exist_ok=True)
        os.makedirs(f'{cfg["save_segP"]}/{patientName}', exist_ok=True)        
        #* 파일 읽기
        _, segdata = rutils.read_by_format(segName, cfg["format_seg"])
        _, imgdata = rutils.read_by_format(imgName, cfg["format_img"])

        #if imgdata.shape != segdata.shape:
            #print('NOT SUIT : ', segName)
            #break
        if cfg["dataset_name"] == "PASeg":
            imgdata = imgdata[ : , : , : , 0]   # [x, y, z, 1] -> [x, y, z]
        
        if segdata.shape[0] == segdata.shape[1]:
            h, w, z = segdata.shape
        elif segdata.shape[0] > segdata.shape[-1]:
            h, w, z = segdata.shape
        else:
            z, h, w = segdata.shape
    
        #! 2d slice로 잘라서 처리
        for _z in range(z):
            if segdata.shape[0] == segdata.shape[1]:
                mask, img = segdata[:,:,_z], imgdata[:,:,_z]
            elif segdata.shape[0] > segdata.shape[-1]:
                mask, img = segdata[:,:,_z], imgdata[:,:,_z]
            else:
                mask, img = segdata[_z], imgdata[_z]
            if mask.shape != img.shape: 
                print('err', imgName, segName, _z)
                break
            if not cutils.checkRatio(mask, h, w):   continue
            #* mask 처리
            is_save_img = save_Mask(mask, cfg["save_segP"], patientName, _z, h, w, cfg["CLASSES_NAME"])
            #* image 처리
            if is_save_img:
                img = cutils.normalizePlanes(img) * 255
                cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}/{_z}.png', img)
    
if __name__ == "__main__":
    root = '/home/nute11a/dataset'
    main(
        config_file = '/home/nute11a/workspace/0_Data_preprocessing/configs/NSCLC Pleural Effusion_1.json'
    )
