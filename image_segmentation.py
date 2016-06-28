from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.filters import roberts
import sys
from skimage.color import label2rgb
from scipy import ndimage as ndi


def segmentation(file_name):
    data_x, data_y, data_z = get_data(file_name)
    shape_x = len(np.unique(data_x))
    shape_y = len(np.unique(data_y))
    X = data_x.reshape(shape_x, shape_y)
    Y = data_y.reshape(shape_x, shape_y)
    Z = data_z.reshape(shape_x, shape_y)

    markers = np.zeros_like(Z)
    markers[Z < 0.1] = 1
    markers[Z > 0.25] = 2
    elevation_map = roberts(Z)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    # ax.imshow(Z)
    # ax.imshow(elevation_map, cmap=plt.cm.jet, interpolation='nearest')
    segmentation = watershed(elevation_map, markers)
    ax2.imshow(segmentation, interpolation='nearest')
    # ax.axis('off')
    # ax.set_title('segmentation')
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_coins, _ = ndi.label(segmentation)
    ax1.imshow(Z, cmap=plt.cm.gray, interpolation='nearest')
    ax1.contour(segmentation, [0.5], linewidths=1.2, colors='y')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')

    plt.show()
    
def get_data(file_name):
    data = np.genfromtxt(file_name, dtype=float, delimiter=None,
                         skip_header=2, names=["Qx", "Qy", "I(Qx,Qy)", "err(I)"])
    shape_x = len(np.unique(data['Qx']))
    shape_y = len(np.unique(data['Qy']))
    data_x = data['Qx']#.reshape(shape_x, shape_y)
    data_y = data['Qy']#.reshape(shape_x, shape_y)
    data_z = data['IQxQy']#.reshape(shape_x, shape_y)
    return data_x, data_y, data_z

def main():
    segmentation("C:/Users/tsy/Documents/Archive/Bio-SANS/anisotropy_with-peak/high-q-has_peak/BioSANS_exp317_scan0021_0001_Iqxy.dat")

if __name__ == "__main__":
    main()
