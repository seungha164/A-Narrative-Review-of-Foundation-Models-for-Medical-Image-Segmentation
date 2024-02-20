import os
import re

def getFileListAll(root):
    fileList = []
    for parenet, _ , files in os.walk(root):
        for f in files:
            if len(re.findall('.gif|.tif|.jpg|.png|.bmp', f)) > 0:  #! format check
                fileList.append(os.path.join(parenet, f))
    return fileList