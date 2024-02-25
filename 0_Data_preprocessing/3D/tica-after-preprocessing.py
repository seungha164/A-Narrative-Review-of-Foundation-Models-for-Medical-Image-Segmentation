from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
import json

import custom_utils as cutils
import read_utils as rutils

def save_Mask_oneCls(mask, saveMaskRoot, saveMaskPatient, z_axis, h, w, clsNum, clsName):
    is_save_img =False
    eq_cls_cnt = 0
    # 해당 cls에 대한 mask 저장
    mask_cls = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     # 0: background
    for con in contours:
        mask_zeros = np.zeros([h, w, 1], dtype=np.uint8)
        cv2.fillPoly(mask_zeros, [con], 255)
        if not cutils.checkRatio(mask_zeros, h, w): continue
        mask_zeros = np.repeat(mask_zeros, 3, axis = 2)
        mask_zeros = mask_zeros.astype(np.float32)
        cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}/{z_axis}_{clsNum}_{clsName}_{eq_cls_cnt}.png', mask_zeros)
        eq_cls_cnt += 1         # 동일 class의 다른 객체들 처리해주기 위한 idx
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
    if cfg['dataset_name'] in ['HCC-TACE-Seg', 'NSCLC_Pleural_Effusion_1', 'NSCLC_Pleural_Effusion_2']:
        return imgName.split('/')[-2]
    else:
        return imgName.split('/')[-1].replace(cfg["format_img"], '').replace('_', '-')
    
def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)

    #* (2) images/labels 얻기
    cases = glob(cfg['caseRoot'])
    filesImg, img_per_segfiles = [], {}
    for case in cases:
        img = sorted(glob(f'{case}/img_*'))[0]
        filesImg.append(img)
        img_per_segfiles[img] = glob(f'{case}/seg_*')
        
    #* (3) save
    for idx in tqdm(range(len(filesImg))):
        imgName = filesImg[idx]
        patientName = getPatientName(imgName, cfg)
        
        os.makedirs(f'{cfg["save_imgP"]}/{patientName}', exist_ok=True)
        os.makedirs(f'{cfg["save_segP"]}/{patientName}', exist_ok=True)        
        #* 파일 읽기
        _, imgdata = rutils.read_by_format(imgName, cfg["format_img"])
        for seg_idx, segfile in enumerate(img_per_segfiles[imgName]):    
            _, segdata = rutils.read_by_format(segfile, cfg["format_seg"])
            if imgdata.shape != segdata.shape:
               print('NOT SUIT : ', segdata)
               break
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
                    print('err', imgName, segfile, _z)
                    break
                if not cutils.checkRatio(mask, h, w):   continue
                #* mask 처리
                clsNum = cfg['CLASSES_NAME_reverse'][segfile.split('/')[-1].replace('.nii.gz', '').replace('seg_', '')]
                clsName = cfg['CLASSES_NAME'][clsNum]
                is_save_img = save_Mask_oneCls(mask, cfg["save_segP"], patientName, _z, h, w, clsNum, clsName)
                #* image 처리
                if is_save_img:
                    img = cutils.normalizePlanes(img) * 255
                    cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}/{_z}.png', img)
            #mask = segdata[:, :, 0]
            #h, w = mask.shape
            #if not cutils.checkRatio(mask, h, w):   continue
    
            
    
            
    
if __name__ == "__main__":
    root = '/home/nute11a/dataset'
    config_root = '/home/nute11a/workspace/SAMM/0_Data_preprocessing/configs'
    main(
        config_file = f'{config_root}/Adrenal-ACC-Ki67-Seg.json'
    )
