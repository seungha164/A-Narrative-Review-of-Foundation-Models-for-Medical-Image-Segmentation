import shutil
from glob import glob
from tqdm import tqdm
import tarfile

TARGET_DIR = '/home/nute11a/dataset/CT/SegRap2023'
FORMAT = 'zip'
DST_DIR = '/home/nute11a/dataset/CT/SegRap2023/data'

targets = glob(f"{TARGET_DIR}/*.{FORMAT}")
for target in tqdm(targets):
    targetName = target.split('/')[-1].replace(f'.{FORMAT}', '')
    #* 압축 풀기
    if FORMAT == 'zip':
        shutil.unpack_archive(target, DST_DIR, FORMAT)
    elif FORMAT == 'tar':
        tar = tarfile.open(target, 'r')
        tar.extractall()
        tar.close()        