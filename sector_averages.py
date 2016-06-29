import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import scipy.stats as stats
from pylab import figure, cm
import sys
from scipy import ndimage
from scipy.optimize import leastsq
from scipy import signal
from scipy import interpolate
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image):

    background = (image >= 0.5)
    eroded_background = binary_erosion(background, border_value=1)
    detected_peaks = eroded_background
    row_num = len(detected_peaks)
    min_num = int(row_num*0.2)
    max_num = int(row_num * 0.8)
    for i in range(len(detected_peaks)):
        for j in range(len(detected_peaks[i])):
                if i < min_num or i >max_num:
                    detected_peaks[i][j] = False
    return detected_peaks

def gaussian(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def radial_average(data_x, data_y, data_z):
    '''
    '''
    angle = np.arctan2(data_y, data_x)
    angle = np.rad2deg(angle)
    # make it integer from 0 to 360
    angle = np.round(angle).astype(int) + 180

    # limit Q range
    q = np.linalg.norm(np.column_stack((data_x, data_y)), axis=1)
    q_condition = q<0.35

    angle_and_intensity_sum = np.bincount(angle[q_condition],
        weights=data_z[q_condition])
    angle_and_intensity_counts = np.bincount(angle[q_condition])

    angle_and_intensity_average = angle_and_intensity_sum / angle_and_intensity_counts.astype(np.float64)
    angle_and_intensity_average = np.nan_to_num(angle_and_intensity_average) # because division by 0
    angle_and_intensity_average = np.tile(angle_and_intensity_average, 2) # duplicates array

    return angle_and_intensity_average[:450]

def get_data(file_name):
    '''
    No reshape in 2d
    '''
    data = np.genfromtxt(file_name, dtype=float, delimiter=None,
                         skip_header=2, names=["Qx", "Qy", "I(Qx,Qy)", "err(I)"])
    shape_x = len(np.unique(data['Qx']))
    shape_y = len(np.unique(data['Qy']))
    data_x = data['Qx']  # .reshape(shape_x, shape_y)
    data_y = data['Qy']  # .reshape(shape_x, shape_y)
    data_z = data['IQxQy']  # .reshape(shape_x, shape_y)
    return data_x, data_y, data_z

def sector_average(data_x, data_y, data_z, n_bins_angle=100, n_bins_radius=50):
    '''
    '''
    angle = np.arctan2(data_y, data_x)
    angle = np.rad2deg(angle)
    # make it integer from 0 to 360
    angle = np.round(angle).astype(int) + 180

    # radius for every pixel
    radius = np.linalg.norm(np.column_stack((data_x, data_y)), axis=1)

    # normalize data to 1
    data_z = (data_z - data_z.min()) / (data_z.max() - data_z.min())

    H, xedges, yedges, binnumber = stats.binned_statistic_2d(angle, radius, data_z,
                                       bins=[n_bins_angle, n_bins_radius],
                                       statistic='mean')

    xedges_width = (xedges[1] - xedges[0])
    xedges_center = xedges[1:] - xedges_width / 2

    yedges_width = (yedges[1] - yedges[0])
    yedges_center = yedges[1:] - yedges_width / 2

    return H, xedges_center, yedges_center


def do_the_job(file_name):
    data_x, data_y, data_z = get_data(file_name)

    # Raw image
    shape_x = len(np.unique(data_x))
    shape_y = len(np.unique(data_y))
    X = data_x.reshape(shape_x, shape_y)
    Y = data_y.reshape(shape_x, shape_y)
    Z = data_z.reshape(shape_x, shape_y)

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax1.pcolormesh(X, Y, Z)

    # Sectors
    H, x, y = sector_average(data_x, data_y, data_z)
    detected_peaks = detect_peaks(H)
    peaks= np.ma.masked_array(H, np.logical_not(detected_peaks))

    ax1.contour(detected_peaks, [0.5], linewidths=1.2, colors='black')
    ax2 = fig.add_subplot(122)
    H_X, H_Y = np.meshgrid(x, y)
    

    ax2.plot(angle_and_intensity_average)


    plt.show()
def main():
    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " Iqxy.dat file")
    else:
        do_the_job(sys.argv[1])

if __name__ == "__main__":
    main()
