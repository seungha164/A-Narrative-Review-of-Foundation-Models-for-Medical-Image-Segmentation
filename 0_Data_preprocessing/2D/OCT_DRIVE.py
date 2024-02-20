from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json

import utils.custom_utils as cutils
import utils.read_utils as rutils

def save_2D_Mask(mask, saveMaskRoot, saveMaskPatient, h, w, clsNames):
    is_save_img = False
    #! Multi-class 처리
    # 1. class 추출
    classes = np.unique(mask).tolist()
    classes.remove(0)           # 0: background
    # 2. 각 class별 mask 저장
    eq_cls_cnt = 0
    for cls in classes:
        mask_cls = cutils.exploit_mask(mask, cls) # 해당 cls의 label만 추출
        #! 동일 class multi-object 처리  
        mask_cls = mask_cls.astype(np.uint8)
        contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for con in contours:
            mask_zeros = np.zeros([h, w, 1], dtype=np.uint8)
            cv2.fillPoly(mask_zeros, [con], 255)
            if not cutils.checkRatio(mask_zeros, h, w): continue
            mask_zeros = np.repeat(mask_zeros, 3, axis = 2)
            mask_zeros = mask_zeros.astype(np.float32)
            if cls == 255:
                cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}_{int(1)}_{clsNames[int(1)]}_{eq_cls_cnt}.png', mask_zeros)
            eq_cls_cnt += 1                # 동일 class의 다른 객체들 처리해주기 위한 idx
            is_save_img = True      # mask 저장시 -> 무조건 img도 저장
    return is_save_img

def getPatientName(imgName, cfg):
    if cfg['dataset_name'] == 'mitoEM':
        dumps = imgName.split('/')
        return f'{dumps[-3]}-{dumps[-1].replace(".png", "")}'
    else:
        return imgName.split('/')[-1].replace(cfg["format"], "")

def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)
    
    #* (2) images/labels 얻기
    filesSeg = glob(cfg["segP"])
    filesImg = [seg.replace('1st_manual', 'images').replace('_manual1.gif', '_training.tif') for seg in filesSeg]
        

    #* (3) 3D -> 2D convert/save
    for idx in tqdm(range(len(filesImg))):
        imgName, segName = filesImg[idx], filesSeg[idx]
        #imgName, segName = './dataset/Microscopy/mitoEM/MitoEM-H/imagesTr/im0421.png', './dataset/Microscopy/mitoEM/MitoEM-H/labelsTr/seg0421.tif'
        patientName = imgName.split('/')[-1].split('_')[0]
        #os.makedirs(f"{cfg['save_imgP']}/{patientName}", exist_ok=True)
        #os.makedirs(f"{cfg['save_segP']}/{patientName}", exist_ok=True)
        #* 파일 읽기
        _, segdata = rutils.read_by_format(segName, cfg["format_seg"])
        _, imgdata = rutils.read_by_format(imgName, cfg["format_img"])
        
        # check
        if imgdata.shape != segdata.shape:
            #imgdata = imgdata[512:-512, 512:-512, :]
            print('NOT SUIT : ', segName)
            break
        
        mask = segdata[:, :, 0]
        h, w = mask.shape
        if not cutils.checkRatio(mask, h, w):   continue
        #* mask 처리
        is_save_img = save_2D_Mask(mask, cfg["save_segP"], patientName, h, w, cfg["CLASSES_NAME"])
        #* image 처리
        if is_save_img:
            cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}.png', imgdata)


if __name__ == "__main__":
    main(
        config_file = './configs/2D/DRIVE.json'
    )