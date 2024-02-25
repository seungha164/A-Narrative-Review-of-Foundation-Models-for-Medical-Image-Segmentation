from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json
import re

import utils.custom_utils as cutils
import utils.read_utils as rutils

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

def getImageFiles(segfiles, ds):
    if (ds == 'COVID-19 Radiography'):
        return [seg.replace('/masks/', '/images/') for seg in segfiles]
    if (ds == 'ICIAR2018'):
        llr = []
        for seg in segfiles:
            dmps = seg.split('/')[-1]
            llr.append(seg[:-len(dmps)].replace('/gt_thumbnails_split2/', '/thumbnails/') + dmps.split('_')[0] + '.png')
        return llr
    if ds == 'UwaterlooSkinCancer':
        return [seg.replace('_contour.png', '_orig.jpg') for seg in segfiles]
    if ds == 'ONETOMANY_TOOLSYNSEG':
        return [seg.replace('/annotations/binary/', '/images/') for seg in segfiles]
    if ds == 'GlaS@MICCAI2015':
        return [seg.replace('_anno.bmp', '.bmp') for seg in segfiles]
    if ds == 'PathologyIMagesForGlandSeg':
        return [seg.replace('/labels/', '/images/') for seg in segfiles]
    if ds == 'BCSS':
        return [seg.replace('/masks/', '/rgbs_colorNormalized_1/') for seg in segfiles]
    
def main(dsroot, config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)
        
    os.makedirs(f'{dsroot}/{cfg["save_imgP"]}', exist_ok=True)
    os.makedirs(f'{dsroot}/{cfg["save_segP"]}', exist_ok=True)
    
    #* (2) images/labels 얻기
    filesSeg = glob(f'{dsroot}/{cfg["segP"]}')
    filesImg = getImageFiles(filesSeg, cfg["dataset_name"])

    #* (3) save
    for idx in tqdm(range(len(filesImg))):
        imgName, segName = filesImg[idx], filesSeg[idx]
        patientName = imgName.split('/')[-1].replace(cfg["format_img"], '').replace('_', '-')
        
        #* 파일 읽기
        _, segdata = rutils.read_by_format(segName, cfg["format_seg"])
        _, imgdata = rutils.read_by_format(imgName, cfg["format_img"])

        #if imgdata.shape != segdata.shape:
            #print('NOT SUIT : ', segName)
            #break

        mask = segdata[:, :, 0]
        h, w = mask.shape
        if not cutils.checkRatio(mask, h, w):   continue
        #* mask 처리
        clss = np.unique(mask)
        for cls in clss:
            is_save_img = save_2D_Mask(mask, f'{dsroot}/{cfg["save_segP"]}', patientName, h, w, cls, cfg["CLASSES_NAME"][cls])
        #* image 처리
        if is_save_img:
            cv2.imwrite(f'{dsroot}/{cfg["save_imgP"]}/{patientName}.png', imgdata)

if __name__ == "__main__":
    root = '/home/nute11a/nfs_server/dataset'
    config2d_p = '/home/nute11a/workspace/SAMM/0_Data_preprocessing/configs'
    main(
        dsroot = root,
        #config_file = './configs/2D/EDD2020.json'
        config_file  = f'{config2d_p}/2D/BCSS.json' 
        #config_file  = './configs/2D/COVID-19 Radiography.json'
    )