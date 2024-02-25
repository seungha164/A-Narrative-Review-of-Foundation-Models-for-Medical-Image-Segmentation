from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json
import re

import utils.custom_utils as cutils
import utils.read_utils as rutils

def mapping_image_per_labelfiles(imgs, cfg):
    dic = {}
    if cfg['dataset_name'] == 'EDD2020':
        for img in imgs:
            name = img.split('/')[-1].replace('.jpg', '')
            dic[img] = glob(f"{cfg['segP']}/{name}*")
    if cfg['dataset_name'] == 'ICIAR2018':
        for img in imgs:
            name = img.split('/')[-1].replace('.png', '').replace('_thumb', '')
            dic[img] = glob(f"{cfg['segP']}/{name}*")
    if cfg['dataset_name'] == 'BSISeg':
        for img in imgs:
           # name = img.split('/')[-1].replace('.png', '').replace('_thumb', '')        
            dic[img] = sorted(glob(f"{img.replace('/image/', '/real_segmentation/').replace('.png', '_*')}"))
    if cfg['dataset_name'] == 'PAPILA':
        for img in imgs:
           # name = img.split('/')[-1].replace('.png', '').replace('_thumb', '')        
            dic[img] = sorted(glob(f"{img.replace('/FundusImages/', '/ExpertsSegmentations/Contours_png/').replace('.jpg', '_*')}"))
    if cfg['dataset_name'] == 'BCSS':
        for img in imgs:
           # name = img.split('/')[-1].replace('.png', '').replace('_thumb', '')        
            dic[img] = sorted(glob(f"{img.replace('/rgbs_colorNormalized_1/', '/masks/')}"))
    if cfg['dataset_name'] == 'WSSS4LUAD':
        for img in imgs:
            dic[img] = sorted(glob(f"{img.replace('/img/', '/mask_split/').replace('.png', '*')}"))
    if cfg['dataset_name'] == 'IDRiD':
        for img in imgs:
            case = img.split('/')[-1].replace('.jpg', '')
            dic[img] = sorted(glob(f"{cfg['segP']}/{case}*"))
    if cfg['dataset_name'] == 'RAVIR':
        for img in imgs:
            case = img.split('/')[-1].replace('.png', '')
            dic[img] = sorted(glob(f"{cfg['segP']}/{case}*"))
    return dic

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

def main(dsroot, config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)
    cfg['save_imgP'] = f'{dsroot}/{cfg["save_imgP"]}'
    cfg['save_segP'] = f'{dsroot}/{cfg["save_segP"]}'
    cfg['imgP']      = f'{dsroot}/{cfg["imgP"]}'
    cfg['segP']      = f'{dsroot}/{cfg["segP"]}'
    
    
    os.makedirs(cfg["save_imgP"], exist_ok=True)
    os.makedirs(cfg["save_segP"], exist_ok=True)
    
    #* (2) images/labels 얻기
    filesImg =  glob(cfg["imgP"])
    img_per_segfiles = mapping_image_per_labelfiles(filesImg, cfg)
    
    #* (3) save
    for idx in tqdm(range(len(filesImg))):
        imgName = filesImg[idx]
        patientName = imgName.split('/')[-1].replace(cfg["format_img"], '').replace('_', '-')
        if cfg['dataset_name'] == 'WSSS4LUAD':
            patientName = imgName.split('/')[-3] + '-' + patientName
        filesSeg = img_per_segfiles[imgName]
        if len(filesSeg) < 1:
            continue
        #* 파일 읽기 
        _, imgdata = rutils.read_by_format(imgName, cfg["format_img"])
        for idx_label, segLabels in enumerate(filesSeg):    #! 각 class 파일별 추출
            #* 파일 읽기
            _, segdata = rutils.read_by_format(segLabels, cfg["format_seg"])
            if imgdata.shape != segdata.shape:
               print('NOT SUIT : ', segLabels)
               break 
            
            mask = np.max(segdata, axis=2)  #segdata[:, :, 0]
            h, w = mask.shape
            if not cutils.checkRatio(mask, h, w):   continue
            #* mask 처리
            if cfg['dataset_name'] == 'BSISeg':
                idx_label = int(segLabels.split('/')[-1].replace('.png', '').split('_')[-1])-1
            is_save_img = save_2D_Mask(mask, cfg["save_segP"], patientName, h, w, idx_label+1, cfg["CLASSES_NAME"][idx_label+1])
            #* image 처리
            if is_save_img:
                cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}.png', imgdata)


if __name__ == "__main__":
    root = '/home/nute11a/nfs_server/dataset'
    config2d_p = '/home/nute11a/workspace/SAMM/0_Data_preprocessing/configs'
    
    main(
        dsroot = root, 
        config_file = f'{config2d_p}/2D/RAVIR.json'
        #config_file = f'{config2d_p}/2D/IDRiD.json'
        #config_file = './configs/2D/EDD2020.json'
        #config_file  = f'{config2d_p}/2D/ICIAR2018.json'
        #config_file  = f'{config2d_p}/2D/BSISeg.json'
    )