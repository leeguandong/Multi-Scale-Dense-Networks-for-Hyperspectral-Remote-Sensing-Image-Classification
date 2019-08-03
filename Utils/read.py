import spectral
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

# img = open_image('6020_3_0.tif')
# input_mat = scio.loadmat('F:/transfer code/Tensorflow  Learning/3D-MSDNet/datasets/IN/Indian_pines_corrected.mat')
# input_mat = input_mat['indian_pines_corrected']

input_mat = scio.loadmat('F:/transfer code/Tensorflow  Learning/3D-MSDNet/datasets/ksc/KSC.mat')
input_mat = input_mat['KSC']
mat_gt = scio.loadmat('F:/transfer code/Tensorflow  Learning/3D-MSDNet/datasets/ksc/KSC_gt.mat')
gt_IN = mat_gt['KSC_gt']

input_mat = input_mat.astype('float32')
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
input_mat = spectral.imshow(input_mat)
plt.savefig('ksc.png')

gt = spectral.imshow(classes=gt_IN)
plt.savefig('gt_ksc.png')
