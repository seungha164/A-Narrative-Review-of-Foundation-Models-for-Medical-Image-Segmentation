from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json

import utils.custom_utils as cutils
import utils.read_utils as rutils
#!! multi-mask(각 png 존재) - one image

def save_2D_Mask(mask, saveMaskRoot, saveMaskPatient, h, w, clsNum, clsName):
    is_save_img =False
    eq_cls_cnt = 0
    #! Multi-class 처리
    mask_cls = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        mask_zeros = np.zeros([h, w, 1], dtype=np.uint8)
        cv2.fillPoly(mask_zeros, [con], 255)
        if not cutils.checkRatio(mask_zeros, h, w): continue
        mask_zeros = np.repeat(mask_zeros, 3, axis = 2)
        mask_zeros = mask_zeros.astype(np.float32)
        cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}_{clsNum}_{clsName}_{eq_cls_cnt}.png', mask_zeros)
        eq_cls_cnt += 1                # 동일 class의 다른 객체들 처리해주기 위한 idx
        is_save_img = True      # mask 저장시 -> 무조건 img도 저장
    return is_save_img

def getPatientName(imgName, cfg):
    if cfg['dataset_name'] == 'mitoEM':
        dumps = imgName.split('/')
        return f'{dumps[-3]}-{dumps[-1].replace(".png", "")}'
    #if cfg['dataset_name'] == 'mitoEM':
        #dumps = imgName. 
    else:
        return imgName.split('/')[-1].replace(cfg["format"], "")

def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)
    
    #* (2) images/labels 얻기
    #filesSeg = glob(cfg["segP"])
    filesImg =  glob(cfg["imgP"])
        

    #* (3) 3D -> 2D convert/save
    for idx in tqdm(range(len(filesImg))):
        imgName = filesImg[idx]
        patientName = imgName.split('/')[-1].replace('.png', '')
        filesSeg = glob(f"{cfg['segP']}/{patientName}.png")
        
        #* 파일 읽기 
        _, imgdata = rutils.read_by_format(imgName, cfg["format"])
        for idx_label, segLabels in enumerate(filesSeg):
            #* 파일 읽기
            _, segdata = rutils.read_by_format(segLabels, cfg["format"])
            if imgdata.shape != segdata.shape:
               print('NOT SUIT : ', segLabels)
               break 
            mask = segdata[:, :, 0]
            h, w = mask.shape
            if not cutils.checkRatio(mask, h, w):   continue
            #* mask 처리
            is_save_img = save_2D_Mask(mask, cfg["save_segP"], patientName, h, w, idx_label+1, cfg["CLASSES_NAME"][idx_label+1])
            #* image 처리
            if is_save_img:
                cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}.png', imgdata)


if __name__ == "__main__":
    main(
        config_file = './configs/2D/DRAC2022.json'
    )