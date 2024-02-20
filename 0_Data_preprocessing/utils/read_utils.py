import nrrd
import nibabel as nib
import SimpleITK as sitk
import cv2

def read_nrrd(p):
    data, header = nrrd.read(p)
    # data      : numpy array
    # header    : OrderedDict
    return header, data

def read_nifti(p):  # .nii.gz
    proxy = nib.load(p)
    header = proxy.header
    data = proxy.get_fdata()
    return header, data

def read_mh(p):  # .mha & .mhd
    itk_image = sitk.ReadImage(p)
    data = sitk.GetArrayViewFromImage(itk_image)
    return 'None', data

def read_2d(p):  # .mha & .mhd
    return 'None', cv2.imread(p)

def read_frames(p):  # .mha & .mhd
    gif = cv2.VideoCapture(p)
    ret, frame = gif.read()
    return ret, frame

def read_by_format(file, format='png'):
    if format in ['.nii.gz', '.nii']:
        return read_nifti(file)
    elif format == '.nrrd':
        return read_nrrd(file)
    elif format in ['.mha', '.mhd']:
        return read_mh(file)
    # 2D
    elif format in ['.png', '.tif', '.jpg']:
        return read_2d(file)
    elif format in ['.gif']:
        return read_frames(file)