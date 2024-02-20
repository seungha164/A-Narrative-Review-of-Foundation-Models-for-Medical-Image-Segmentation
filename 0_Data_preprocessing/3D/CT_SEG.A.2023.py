import numpy as np
import SimpleITK as sitk
import nrrd
from glob import glob
import cv2
import os
from tqdm import tqdm
import shutil
import read_utils as rutils

def checkRatio(arr, h, w):
    ratio = (np.count_nonzero(arr) / (h * w)) * 100
    if ratio < 0.153:
        return False
    else:
        return True

def exploit_mask(mask, pixel):
    m = mask.copy()
    m[m != pixel] = 0
    m[m == pixel] = 255
    return m

# 1. 모든 파일 불러오기

nrrdSegP = '../dataset/CT/SEG.A.2023/*/*/*.seg.nrrd'
filesSeg = glob(nrrdSegP)
CLASSES_NAME = ['background', 'AorticVesselTree']
# 2. nrrd로 각 파일 읽기
for segName in tqdm(filesSeg):
    # save 경로 설정
    savepath = segName[:-len(segName.split('/')[-1])-1]
    if os.path.isdir(f"{savepath}/images"):
        shutil.rmtree(f"{savepath}/images")
    if os.path.isdir(f"{savepath}/masks"):
        shutil.rmtree(f"{savepath}/masks")
    os.makedirs(f"{savepath}/images", exist_ok=True)
    os.makedirs(f"{savepath}/masks", exist_ok=True)
    # anno / img 읽기
    imgName = segName.replace('seg.nrrd', 'nrrd')
    # 1. nrrd로 읽기
    _, imgdata = rutils.read_nrrd(imgName)
    _, segdata = rutils.read_nrrd(segName)
    # check
    if imgdata.shape != segdata.shape:
        print('NOT SUIT : ', segName)
        break
    #! 2D img로 보기 - z divide
    h, w, z = segdata.shape
    for _z in range(z):
        img = segdata[:,:,_z]
        if not checkRatio(img, h, w): continue
        # mask 처리 - 객체 분리
        #! Multi-class인 경우 처리
        classes = np.unique(img).tolist()
        classes.remove(0)  # 0: background
        is_save_img = False
        for cls in classes:
            # 해당 cls의 label만 추출
            img_cls = exploit_mask(img, cls)
            #! 각 class별 multi-object 처리
            contours, _ = cv2.findContours(img_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            eq_cls_cnt = 0
            for con in contours:
                mask_zeros = np.zeros([h, w, 1], dtype=np.uint8)
                cv2.fillPoly(mask_zeros, [con], 255)
                if not checkRatio(mask_zeros, h, w): continue
                mask_zeros = np.repeat(mask_zeros, 3, axis = 2)
                cv2.imwrite(f'{savepath}/masks/{_z}_{cls}_{CLASSES_NAME[cls]}_{eq_cls_cnt}.png', mask_zeros)
                eq_cls_cnt += 1                # 동일 class의 다른 객체들 처리해주기 위한 idx
                is_save_img = True      # mask 저장시 -> 무조건 img도 저장
        # 이미지 처리
        if is_save_img:
            cv2.imwrite(f'{savepath}/images/{_z}.png', imgdata[:,:,_z])