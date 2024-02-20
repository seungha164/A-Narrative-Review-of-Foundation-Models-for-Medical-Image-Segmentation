import os
import argparse
from PIL import Image
from tqdm import tqdm
import json
import shutil
import re
import myUtils
import cv2

#EDD_CONFIG = {'BE' : 1, 'cancer' : 2, 'HGD' : 3, 'polyp' : 4, 'suspicious' : 5}
SEGPC_CONFIG = {1 : 'Cytoplasm', 2 : 'Nucleus'}
PAPILA_CONFIG = {'cup' : 1, 'disc' : 2}
RBISDDSM_CONFIG = {'calc_ben_mask': 1, 'calc_mlgn_mask': 2, 'mass_ben_mask' : 3, 'mass_mlgn_mask' : 4}
MONUSAC2020_CONFIG = {'Epithelial' : 1, 'Lymphocyte' : 2, 'Macrophage': 3, 'Neutrophil': 4, 'Ambiguous': 5}
def exploit_raw_name(sample, dataset):
    #if sample == '/home/nute11a/dataset/MR/SMRA2021/labelsTr/P789/342-ICAL_1_Lumen_0.png':
        #print()
    #if dataset == 'UWaterlooSkinCancer':
        #return sample.replace('_contour.png', '').replace('_orig.jpg', '')
    if dataset == 'DRIVE':
        if 'mask' in sample: return False   # mask가 아니라 mannual 사용
        return sample.split('/')[-1].split('_')[0]  # 마지막만 떼기
    elif dataset in ['PAPILA', 'ONETOMANY_TOOLSYNSEG', 'UWaterlooSkinCancer', 'DRAC2022', 'Kvasir-SEG', 'EDD2020', 'ICIAR2018', 'mitoEM', 'BSISeg', 'COVID-19 Radiography']:       #! 정리한 2D dataset들
        if 'imagesTr_png' not in sample and 'labelsTr_png' not in sample: return False
        return f"{sample.split('/')[-1].replace('.png', '').split('_')[0]}"
    elif dataset == 'COVID-19 Radiography':
        return sample.split('/')[-1].split('.')[0]  # 마지막만 떼기
    elif dataset == 'RAVIR':
        return sample.split('/')[-1].split('.')[0].replace('_1_Artery', '').replace('_2_Vein', '')  # 마지막만 떼기
    elif dataset == 'IDRiD':
        return sample.split('/')[-1].split('.')[0].replace('_MA', '').replace('_HE', '').replace('_EX', '').replace('_SE', '').replace('_OD', '')
    #elif dataset == 'EDD2020':
    #    if '/masks/' in sample: return False   # mask가 아니라 mannual 사용
    #    return sample.split('/')[-1].split('.')[0].replace('_BE', '').replace('_cancer', '').replace('_HGD', '').replace('_polyp', '').replace('_suspicious', '')  # 마지막만 떼기
    elif dataset == 'SEGPC2021':
        return sample.split('/')[-1].split('.')[0].split('_')[0]
    #elif dataset == 'ICIAR2018':
    #    if ('/WSI/' not in sample) or ('/gt_thumbnails/' in sample): return False   # WSI(whole-slide)만 사용
    #    return sample.split('/')[-1].split('.')[0].replace('_thumb', '').replace('_1_Benign', '').replace('_2_InSituCarcinoma', '').replace('_3_InvasiveCarcinoma', '')
    #elif dataset == 'ONETOMANY_TOOLSYNSEG':
    #    if 'harmonized_image' in sample:    return False
    #    dumps = sample.split('/')
    #    return f'{dumps[7]}_{dumps[-1].split(".")[0]}'  # 마지막만 떼기
    #elif dataset == 'PAPILA':
    #    return sample.split('/')[-1].split('.')[0].replace('_cup_exp1', '').replace('_disc_exp1', '')
    elif dataset == 'PathologyIMagesForGlandSeg':
        if '/labels/' in sample: return False   # labels_split 사용 예정
        return sample.split('/')[-1].split('.')[0].split('_')[0]
    elif dataset == 'GlaS@MICCAI2015':
        if '/imgs_and_masks/' in sample and 'anno_' in sample: return False # imgs_and_masks_split에 있는 anno만 사용
        return sample.split('/')[-1].split('.')[0].split('_anno')[0]
    elif dataset == 'WSSS4LUAD':
        if ('/mask/' in sample) or  ('/background-mask/' in sample): return False   # mask_split에 있는 anno만 사용
        dumps = sample.split('/')
        return dumps[-3] + '_' +dumps[-1].split('.')[0].replace('_1_Tumor', '').replace('_2_Stroma', '').replace('_3_Normal', '')
    elif dataset == 'ARCADE':
        if ('/annotations/') in sample: return False
        dumps = sample.split('/')
        return f"{dumps[6]}_{dumps[7]}_{dumps[-1].split('.')[0].split('_')[0]}"
    elif dataset == 'RBIS-DDSM':
        if ('_split/' not in sample) and ('/images/' not in sample):  return False
        dumps = sample.split('/')[-1].split('.')[0].split('_')
        return f"{dumps[0]}_{dumps[1]}_{dumps[2]}_{dumps[3]}"
    elif dataset == 'BCSS':
        if '/masks/' in sample: return False
        return f"{sample.split('/')[-1].split('.')[0]}.2500"
    elif dataset == 'MoNuSAC2020':
        if 'MoNuSAC_Testing_Color_Coded_Masks' in sample: return False
        if '/MoNuSAC_masks/' in sample:
            return sample.split('/')[9]
        else:
            return sample.split('/')[-1].split('.')[0]
    elif dataset == 'VESSEL12':
        if '/annotations2/' not in sample and '/data2/' not in sample: return False
        dumps = sample.split('/')
        return f'{dumps[-2]}-{dumps[-1].replace(".png", "").split("_")[0]}'
    elif dataset == 'SEG.A.2023':
        dumps = sample.split('/')
        return f"{dumps[-3]}-{dumps[-1].replace('.png', '').split('_')[0]}"
    elif dataset == 'MSD-Brain':    # 4D
        dumps = sample.split('/')
        base = dumps[-1].replace('.png', '').split('_')
        return f"{dumps[-2]}-{base[0]}_{base[1]}"
    elif ('MSD-' in dataset) or dataset in ['PleThora', 'InnerEarSeg', 'Shifts2022', 'SMRA2021', 'CrossModa2021', 'CrossModa2022', 'SPIDER', 'ATLAS2023', 'PASeg', 'TDSC-ABUS2023', 'RESECT(US)', 'LiTS', 'BrainPTM2021']:
        dumps = sample.split('/')
        return f"{dumps[-2]}-{dumps[-1].replace('.png', '').split('_')[0]}"
    elif dataset == 'SegRap2023':
        dumps = sample.split('/')
        return f"{dumps[-4]}-{dumps[-2]}-{dumps[-1].replace('.png', '').split('_')[0]}"

def matching_img_mask_pair(samples, dataset):
    result = {}
    
    for sample in samples:
        raw_name = exploit_raw_name(sample, dataset)    #! 중요
        if not raw_name:    continue    # 사용 안할 파일들은 pass

        if raw_name in result:
            result[raw_name].append(sample)
        else:
            result[raw_name] = [sample]
    # 매칭 안된거는 다 삭제
    notMatchingIdx = [key for key in result if len(result[key]) < 2]    # <2 : 매칭 X   | 2 : img-mask(binary)  | 2> : img-masks(multi-class)
    for i in notMatchingIdx:
        del result[i]
    return result

def save_iamgs_masks(pairs, saveRoot, dataset):
    labele2image, mappings = {}, {}
    # images / masks 폴더 생성
    saveRoot_img, saveRoot_mask = os.path.join(saveRoot, 'images'), os.path.join(saveRoot, 'masks')
    os.makedirs(saveRoot_img, exist_ok=True)
    os.makedirs(saveRoot_mask, exist_ok=True)
    for idx, key in enumerate(tqdm(pairs)):
        img, masks, cls_per_anno = '', [], {}
        mappings[idx] = pairs[key]
        for sample in pairs[key]:
            # 0. 저장 경로 설정
            #if dataset == 'UWaterlooSkinCancer':
            #    saveP = os.path.join((saveRoot_img if ('orig.' in sample) else saveRoot_mask), (str(idx) + '.png'))
            #if dataset in ['DRIVE', 'ONETOMANY_TOOLSYNSEG']:
            #    saveP = os.path.join((saveRoot_img if ('images' in sample) else saveRoot_mask), (str(idx) + '.png'))
            #elif dataset == 'DRAC2022':
            #    if '1. Original Images' in sample:
            #        saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
            #    else:   # multi label
            #        label = sample.split('/')[-2].replace('. ', '_').replace(' ', '')
            #        saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{label}.png')
            if dataset == 'RAVIR':
                if 'training_images' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:   # multi label
                    _, _, _, classNum, className = sample.replace('.png', '').split('/')[-1].split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{classNum}_{className}.png')
            elif dataset == 'IDRiD':
                if '1. Original Images' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    classNum, className = sample.split('/')[-2].split('. ')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{classNum}_{className.replace(" ", "")}.png')
            #elif dataset == 'EDD2020':
            #    if 'originalImages' in sample:
            #        saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
            #    else:
            #        className = sample.split('/')[-2]
            #        saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{EDD_CONFIG[className]}_{className}.png')
            elif dataset == 'SEGPC2021':
                if '/x/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, num_of_anno, classNum = sample.split('/')[-1].split('_') # 한 이미지에 같은 class label이 다수
                    classNum = 1 if int(classNum[:2]) == 20 else 2
                    # SEGPC_CONFIG = {1 : 'Cytoplasm', 2 : 'Nucleus'}
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{classNum}_{SEGPC_CONFIG[classNum]}_{num_of_anno}.png')
            #elif dataset == 'ICIAR2018':
            #    if '/thumbnails/' in sample:
            #        saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
            #    else:
            #        _, clsNum, clsName = sample.split('/')[-1].split('.')[0].split('_')
            #        saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}.png')
            #elif dataset == 'PAPILA':
            #    if '/FundusImages/' in sample:
            #        saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
            #    else:
            #        clsName = sample.split('/')[-1].split('_')[1]
            #        saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{PAPILA_CONFIG[clsName]}_{clsName}.png')
            elif dataset == 'PathologyIMagesForGlandSeg':
                if '/images/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, clsNum, clsName, anno = sample.split('/')[-1].split('.')[0].split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{anno}.png')
            elif dataset == 'GlaS@MICCAI2015':
                if '/imgs_and_masks/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, _, _, clsNum, clsName, anno = sample.split('/')[-1].split('.')[0].split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{anno}.png')
            elif dataset == 'WSSS4LUAD':
                if '/img/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, clsNum, clsName = sample.split('/')[-1].split('.')[0].split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}.png')
            elif dataset == 'ARCADE':
                if '/images/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, clsNum, clsName, anno = sample.split('/')[-1].split('.')[0].split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{anno}.png')
            elif dataset == 'RBIS-DDSM':
                if '/images/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    dumps = sample.split('/')
                    clsName = dumps[-2].replace('_split', '')
                    anno = dumps[-1].split('.')[0].split('_')[-1]
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{RBISDDSM_CONFIG[clsName]}_{clsName.replace("_", "")}_{anno}.png')
            elif dataset == 'BCSS':
                if '/rgbs_colorNormalized_1/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, clsNum, clsName, _ = sample.split('/')[-1].split('.')[1].split('_')
                    if clsName in cls_per_anno:
                        cls_per_anno[clsName] += 1
                    else:
                        cls_per_anno[clsName] = 0
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{cls_per_anno[clsName]}.png')
            elif dataset == 'MoNuSAC2020':
                if '/MoNuSAC_images_and_annotations/' in sample or '/MoNuSAC Testing Data and Annotations/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    clsName = sample.split('/')[-2]
                    if clsName in cls_per_anno:
                        cls_per_anno[clsName] += 1
                    else:
                        cls_per_anno[clsName] = 0
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{MONUSAC2020_CONFIG[clsName]}_{clsName}_{cls_per_anno[clsName]}.png')
            elif dataset == 'VESSEL12':
                if 'data2' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    cnt = sample.split('/')[-1].replace('.png', '').split('_')[-1]
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_1_Lung_{cnt}.png')
            elif dataset == 'SEG.A.2023':
                if 'images' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, clsNum, clsName, cnt = sample.split('/')[-1].replace('.png', '').split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{cnt}.png')
            elif dataset == 'MSD-Brain':
                if '/imagesTr_png/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:   # 4d => view(ex. T1, flare)
                    _, _, clsNum, clsName, cnt = sample.split('/')[-1].replace('.png', '').split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{cnt}.png')
            elif ('MSD-' in dataset) or dataset in ['PleThora', 'PAPILA', 'ONETOMANY_TOOLSYNSEG', 'UWaterlooSkinCancer', 'COVID-19 Radiography', 'BSISeg', 'ICIAR2018', 'EDD2020', 'Kvasir-SEG', 'DRAC2022', 'InnerEarSeg', 'mitoEM', 'Shifts2022', 'SMRA2021', 'CrossModa2021', 'CrossModa2022', 'SPIDER', 'ATLAS2023', 'PASeg', 'TDSC-ABUS2023', 'RESECT(US)', 'LiTS', 'SegRap2023', 'BrainPTM2021']:
                if '/imagesTr_png/' in sample:
                    saveP = os.path.join(saveRoot_img, (str(idx) + '.png'))
                else:
                    _, clsNum, clsName, cnt = sample.split('/')[-1].replace('.png', '').split('_')
                    saveP = os.path.join(saveRoot_mask, f'{str(idx)}_{clsNum}_{clsName}_{cnt}.png')
            
            # 1-1. png로 통일하여 저장
            if '.png' not in sample:    # format 검사 - png
                im = Image.open(sample).convert('RGB')
                im.save(saveP, 'png')
            else:
                shutil.copyfile(sample, saveP)
            # 1-2. 후처리 코드
            if dataset == 'IDRiD' and '1. Original Images' not in sample:  # [0,0,255] -> [255,255,255]
                mask = cv2.imread(saveP)
                mask[mask[:,:,2] == 255] = [255,255,255]
                cv2.imwrite(saveP, mask)
            # ++
            if saveRoot_img in saveP:    
                img = saveP
            else:
                masks.append(saveP)
        for mask in masks:
            labele2image[mask] = img
    print(f'# of images : {len(os.listdir(saveRoot_img))}')
    print(f'# of masks : {len(os.listdir(saveRoot_mask))}')
    return labele2image, mappings


#!------------------------------------------------
parser = argparse.ArgumentParser(description='[SAM-Med2D] 전처리')
parser.add_argument('--root', default= '/home/nute11a/dataset')
parser.add_argument('--modality', default= 'CT')
parser.add_argument('--dsName', default= 'PleThora')
args = parser.parse_args()

#!------------------------------------------------

dsroot      = os.path.join(args.root, args.modality, 'original', args.dsName)
saveroot    = os.path.join(args.root, args.modality, 'SAM-Med2D',(args.dsName + '_pre'))
os.makedirs(saveroot, exist_ok=True)

print(f'[{args.modality}]  {args.dsName  }')
# 1. 파일 모두 얻기
files = myUtils.getFileListAll(dsroot)
# 2. mask와 conoutr 짝짓기
matchings = matching_img_mask_pair(files, dataset = args.dsName)
# 3. 새롭게 정렬 - '파일 이동' => (datasetName)_pre / masks or images 에 저장
labele2image, mapping_information = save_iamgs_masks(matchings, saveroot, dataset = args.dsName)
with open(os.path.join(saveroot, f'label2image_{args.dsName}.json'), 'w') as jsonF:
    json.dump(labele2image, jsonF, indent=2)
with open(os.path.join(saveroot, f'mappings_{args.dsName}.json'), 'w') as jsonF:
    json.dump(mapping_information, jsonF, indent=2)
