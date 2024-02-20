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


def saveSingleClassMask(mask, saveMaskRoot, saveMaskPatient, z_axis, h, w, clsNum, clsName): 
    is_save_img = False
    #! 동일 class multi-object 처리
    eq_cls_cnt = 0
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        mask_zeros = np.zeros([h, w, 1], dtype=np.uint8)
        cv2.fillPoly(mask_zeros, [con], 255)
        if not cutils.checkRatio(mask_zeros, h, w): continue
        mask_zeros = np.repeat(mask_zeros, 3, axis = 2)
        mask_zeros = mask_zeros.astype(np.float32)
        cv2.imwrite(f'{saveMaskRoot}/{saveMaskPatient}/{z_axis}_{clsNum}_{clsName}_{eq_cls_cnt}.png', mask_zeros)
        eq_cls_cnt += 1                # 동일 class의 다른 객체들 처리해주기 위한 idx
        is_save_img = True      # mask 저장시 -> 무조건 img도 저장
            
    return is_save_img

def normalizePlanes(npzarray):
        """
        Normalizing the image using the appropriate maximum and minimum values associated 
        with a CT scan for lung cancer (in terms of Hounsfeld Units)
        
        """
        max_hu = np.max(npzarray)
        min_hu= np.min(npzarray)#-1000.
        npzarray = (npzarray - min_hu) / (max_hu - min_hu)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
def main(config_file):
    #* (1) read data config file
    with open(config_file, 'rb') as f:
        cfg = json.load(f)
    
    #* (2) images/labels 얻기
    filesImg = glob(cfg["imgP"])
    #filesSeg = glob(cfg["segP"])
    #filesImg = getImageFiles(filesSeg, cfg["dataset_name"])

    #* (3) 3D -> 2D convert/save
    for idx in tqdm(range(len(filesImg))):
        imgName = filesImg[idx]
        segNames = []
        for cls in ['CST_left', 'CST_right', 'OR_left', 'OR_right']:
            segNames +=  glob(imgName.replace('T1', cls))
        if len(segNames) == 0:
            continue
        patientName = imgName.split('/')[-2]
        os.makedirs(f"{cfg['save_imgP']}/{patientName}", exist_ok=True)
        os.makedirs(f"{cfg['save_segP']}/{patientName}", exist_ok=True)
        
        #* 이미지 파일 읽기
        _, imgdata = rutils.read_by_format(imgName, cfg["format"])
        #* seg files 읽기   - 각 class별 파일 존재
        for segName in segNames:
            clsName = segName.split('/')[-1].replace('.nii.gz', '')
            _, segdata = rutils.read_by_format(segName, cfg["format"])
            # check
            if imgdata.shape != segdata.shape:
                print('NOT SUIT : ', segName)
                break
            h, w, z = segdata.shape
            #! 2D img로 보기 - z divide
            for _z in range(z):
                mask, img = segdata[:,:,_z], imgdata[:,:,_z]
                if not cutils.checkRatio(mask, h, w):   continue
                #* mask 처리
                clsNum, clsName_refine = cfg["CLASSES_NAME"][clsName], clsName.replace('_', '-')
                is_save_img = saveSingleClassMask(mask, cfg["save_segP"], patientName, _z, h, w, clsNum, clsName_refine)
                #* image 처리
                if is_save_img:
                    img = normalizePlanes(img) * 255
                    cv2.imwrite(f'{cfg["save_imgP"]}/{patientName}/{_z}.png', img)

if __name__ == "__main__":
    main(
        config_file = './configs/BrainPTM2021.json'
    )