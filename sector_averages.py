import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pylab import figure, cm
import sys
from scipy import ndimage
from scipy import signal
from scipy import interpolate
import scipy.optimize as opt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.ndimage import filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def get_max(data):
    background = (data >= 2*np.mean(data))
    roi = binary_erosion(background, border_value=1)
    row_num = len(roi)
    min_num = int(row_num*0.2)
    max_num = int(row_num * 0.8)

    for i in range(len(roi)):
        for j in range(len(roi[i])):
                if i < min_num or i >max_num:
                    roi[i][j] = False
    max_length = 0
    max_col = 0
    for j in range (len(roi[0])):
        for i in range(len(roi)):
            if roi[i][j] == True:
                last_true = 0
                for length in range(i + 1 , len(roi)):
                    if roi[length][j ]==True and length > last_true:
                        last_true = length
                if (last_true - i) > max_length:
                    max_length  = last_true - i
                    max_col = j
                elif (last_true - i) == max_length:
                    max_col = float((max_col + j)/2)

    return max_col

def detect_peaks(image):
    background = (image >=2 * np.mean(np.nan_to_num(image)))
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
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

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

    fig = plt.figure(figsize = (20, 15))
    ax1 = fig.add_subplot(221)
    ax1.pcolormesh(X, Y, Z)

    # Sectors
    H, x, y = sector_average(data_x, data_y, data_z)
    X_1, Y_1 = np.meshgrid(x, y)

    com = get_max(np.nan_to_num(H))
    print(com)
    # detected_peaks = detect_peaks(H)
    # peaks= np.ma.masked_array(H, np.logical_not(detected_peaks))
    # peaks = np.nan_to_num(peaks)
    # com = ndimage.center_of_mass(peaks)[1]
    # print(com)


    ax2 = fig.add_subplot(222)

    ax2.contourf(X_1, Y_1, H.T, 150)
    ax4 = fig.add_subplot(224)
    row = H[:,int(com)]
    x_axis = np.linspace(0, len(H)/50 * 180, len(H))
    y_ax2 = np.empty(360)
    y_ax2.fill(com/H.shape[1])
    x_ax2 = np.linspace(0, 360, 360)
    ax2.plot(x_ax2, y_ax2)

    row = np.nan_to_num(row)
    row = (row - row.min()) / (row.max() - row.min()) # normalized
    ax4.plot(x_axis, row)

    n_bins = 50
    bin_means, bin_edges, binnumber = stats.binned_statistic(x_axis, row, statistic='mean', bins=n_bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    # normalize to 1
    bin_means = (bin_means - bin_means.min()) / (bin_means.max() - bin_means.min())


    xs = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    rbf = interpolate.Rbf(bin_centers, bin_means)
    spline = rbf(xs)

    popt, pcov = curve_fit(gaussian,xs, spline, p0 = [1, 180, 10])

    ax4.plot(x_axis,gaussian(x_axis,*popt),'ro:',label='fit')
    FWHM = 2*np.sqrt(2*np.log(2))*np.absolute(popt[2])
    sigma_2 = 2*popt[2]
    print("FWHM: ", np.absolute(FWHM ))
    print("2*sigma: ", np.absolute(sigma_2))
    print("center: ", popt[1])
    ax3 = fig.add_subplot(223)
    ax3.pcolormesh(X, Y, Z)
    x_line = np.linspace(np.amin(X),np.amax(X), 100)
    y_line = x_line* np.tan(np.deg2rad(sigma_2/2))
    ax3.plot(x_line, y_line)
    ax3.plot(x_line, -y_line)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0] + " Iqxy.dat file")
    else:
        do_the_job(sys.argv[1])

if __name__ == "__main__":
    main()
