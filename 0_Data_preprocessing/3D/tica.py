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
            cls = 1
            if type(clsNames) == 'dict':
                cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}/{z_axis}_{int(cls)}_{clsNames[int(cls)]}_{eq_cls_cnt}.png', mask_zeros)
            else:
                cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}/{z_axis}_{int(cls)}_{clsNames[int(cls)]}_{eq_cls_cnt}.png', mask_zeros)
            eq_cls_cnt += 1                # 동일 class의 다른 객체들 처리해주기 위한 idx
            is_save_img = True      # mask 저장시 -> 무조건 img도 저장
    return is_save_img


def seg2imgs(segfiles, ds, imgroot):
    dicc = {}
    if ds == 'NSCLC_Pleural_Effusion_1':
        for seg in segfiles:
            case = seg.split('/')[-2]
            dcms = glob(f'{imgroot}/{case}/*/*/*')
            dicc[seg] = dcms
        return dicc

def getPatientName(sefFile, ds):
    if ds == 'NSCLC_Pleural_Effusion_1':
        return sefFile.split('/')[-2]
    
def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)

    #* (2) images/labels 얻기
    filesSeg = glob(cfg["segP"])
    filesImg = seg2imgs(filesSeg, cfg["dataset_name"], cfg["imgP"])


    #* (3) save
    for idx in tqdm(range(len(filesImg))):
        # 1. segfile 부터 읽기
        segName = filesSeg[idx]
        patientName = getPatientName(segName, cfg["dataset_name"])
        #imgName, segName = , filesSeg[idx]
       
        os.makedirs(f'{cfg["save_imgP"]}/{patientName}', exist_ok=True)
        os.makedirs(f'{cfg["save_segP"]}/{patientName}', exist_ok=True)        
        #* 파일 읽기
        _, segdata = rutils.read_by_format(segName, cfg["format_seg"])  # [512, 512, 88]
        #_, imgdata = rutils.read_by_format(imgName, cfg["format_img"])

        #if imgdata.shape != segdata.shape:
            #print('NOT SUIT : ', segName)
            #break
       
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
