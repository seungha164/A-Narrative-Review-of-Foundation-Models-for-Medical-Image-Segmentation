import nrrd
import nibabel as nib
import SimpleITK as sitk

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

def read_by_format(file, format='png'):
    if format in ['.nii.gz', '.nii']:
        return read_nifti(file)
    elif format == '.nrrd':
        return read_nrrd(file)
    elif format in ['.mha', '.mhd']:
        return read_mh(file)