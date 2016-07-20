import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import time
from skimage import img_as_ubyte
from scipy import ndimage
from scipy import signal
from scipy import interpolate
from numba import jit


@jit(nopython=True)
def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

@jit(nopython=True)
def get_mask(image, bool_mask):
    for i in range(len(image)):
        for j in range(len(image[i])):
                if j < 2:
                    bool_mask[i][j] = False
    return bool_mask

def get_max(image):
        copy = image.copy()
        bool_mask = np.ones((image.shape[0], image.shape[1]), dtype = bool)
        bool_mask = get_mask(image, bool_mask)
        bool_mask = img_as_ubyte(bool_mask)
        center = cv2.bitwise_and(image,image,mask = bool_mask)

        orig = center.copy()
        # orig = image.copy()
        gray = cv2.GaussianBlur(np.nan_to_num(orig), (3, 3), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        return maxLoc

def get_data(file_name):
    data = np.genfromtxt(file_name, dtype=float, delimiter=None,
                         skip_header=2, names=["Qx", "Qy", "I(Qx,Qy)", "err(I)"])
    shape_x = len(np.unique(data['Qx']))
    shape_y = len(np.unique(data['Qy']))
    data_x = data['Qx']  # .reshape(shape_x, shape_y)
    data_y = data['Qy']  # .reshape(shape_x, shape_y)
    data_z = data['IQxQy']  # .reshape(shape_x, shape_y)
    return data_x, data_y, data_z


def sector_average(data_x, data_y, data_z, n_bins_angle=100, n_bins_radius=50, max_radius = np.inf):
    radius = np.linalg.norm(np.column_stack((data_x, data_y)), axis=1)

    angle = np.arctan2(data_y, data_x)
    angle = np.rad2deg(angle)
    # make it integer from 0 to 360
    angle = np.round(angle).astype(int) + 180

    angle = angle[radius <= max_radius]
    # radius for every pixel

    # normalize data to 1
    data_z = (data_z - data_z.min()) / (data_z.max() - data_z.min())
    data_z =data_z[radius <= max_radius]
    radius = radius[radius <= max_radius]

    H_orig, xedges, yedges, binnumber = stats.binned_statistic_2d(angle, radius, data_z,
                                       bins=[n_bins_angle, n_bins_radius],
                                       statistic='mean')

    xedges_width = (xedges[1] - xedges[0])
    xedges_center = xedges[1:] - xedges_width / 2

    yedges_width = (yedges[1] - yedges[0])
    yedges_center = yedges[1:] - yedges_width / 2

    H = np.tile(H_orig, (2,1))
    xedges_center = np.linspace(np.amin(xedges_center), 2*np.amax(xedges_center), 2*len(xedges_center))

    return H, H_orig, xedges_center, yedges_center

def data_fit(data, com, H):
    n_bins = 50
    x_axis = np.linspace(0, 720, len(H))
    bin_means, bin_edges, binnumber = stats.binned_statistic(x_axis, data, statistic='mean', bins=n_bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    # normalize to 1
    bin_means = (bin_means - bin_means.min()) / (bin_means.max() - bin_means.min())

    xs = np.linspace(bin_centers.min(), bin_centers.max(), 720)
    rbf = interpolate.Rbf(bin_centers, bin_means)
    spline = rbf(xs)

    radius = 2
    data_split = np.split(spline, 4)[0]
    minima= signal.argrelextrema(data_split, np.less, order=radius)[0]
    dip = minima[0]
    end = dip + 360
    peak = np.argmax(data_split[dip:])
    popt, pcov = opt.curve_fit(gaussian, xs, spline, p0 = [1, peak + 30, 13])

    fit = gaussian(x_axis,*popt)
    FWHM = 2*np.sqrt(2*np.log(2))*np.absolute(popt[2])
    sigma_2 = 2*popt[2]
    center = popt[1]

    popt_1, pcov_1 = opt.curve_fit(gaussian, xs, spline, p0 = [1, center + 180, 13])
    fit_1 = gaussian(x_axis, *popt_1)
    sigma_2_1 = 2*popt_1[2]
    center_1 = popt_1[1]
    rotation = (np.absolute(180-center)+np.absolute(360-center_1))/2
    print("1st peak: %f degrees" % (center))
    print("2nd peak: %f degrees" % (center_1))
    print("difference between peaks: %f degrees" % (center_1- center))
    print("average angle: %f degrees" %((sigma_2 + sigma_2_1)/2))
    print("average rotation: %f degrees" % (rotation))

    return fit, sigma_2, fit_1, sigma_2_1, rotation

def display_graphs(file_name, n_bins_radius=50, max_radius=np.inf):
    start = time.time()
    if max_radius != np.inf:
        max_radius = float(max_radius)
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
    # get_center_of_mass(X, Y, Z)
    # Sectors
    H, H_orig, x, y= sector_average(data_x, data_y, data_z,
                                    n_bins_radius=int(n_bins_radius),
                                    max_radius=(max_radius))
    X_1, Y_1 = np.meshgrid(x, y)
    com = get_max(np.nan_to_num(H_orig))
    ax2 = fig.add_subplot(222)
    ax2.contourf(X_1, Y_1, H.T, 150)
    row = H[:,int(com[0])]
    x_axis = np.linspace(0, 720, len(H))
    y_ax2 = np.empty(720)
    y_ax2.fill(com[0]*(Y_1[2][0] - Y_1[1][0]))
    x_ax2 = np.linspace(0, 720, 720)
    ax2.plot(x_ax2, y_ax2, color = 'w') # Graphs horizontal line

    row = np.nan_to_num(row)
    row = (row - row.min()) / (row.max() - row.min()) # normalized
    fit, sigma_2, fit_1, sigma_2_1, rotation = data_fit(data=row, com=com, H=H)

    ax4 = fig.add_subplot(224)
    ax4.plot(x_axis, row)
    ax4.plot(x_axis,fit ,'ro:',label='fit')
    ax4.plot(x_axis, fit_1, 'ro:')

    x_line = np.linspace(X.min(),X.max(), 100)
    y0_line = x_line* np.tan(np.deg2rad(rotation))
    y_line = x_line* np.tan(np.deg2rad(rotation+ (sigma_2 + sigma_2_1)/4))
    ax3 = fig.add_subplot(223)
    ax3.pcolormesh(X, Y, Z)
    ax3.plot(x_line, y0_line, 'w--')
    ax3.plot(x_line, y_line, 'w')
    ax3.plot(x_line, -y_line, 'w')  # Draws 'X'
    axes = plt.gca()
    axes.set_xlim([X.min(),X.max()])
    axes.set_ylim([Y.min(),Y.max()])
    end = time.time()
    print(end - start)
    plt.show()
def main():
    if len(sys.argv) > 4:
        print("Too many parameters.")
    elif len(sys.argv) == 2:
        display_graphs(sys.argv[1])
    elif len(sys.argv)==3:
        display_graphs(sys.argv[1], sys.argv[2])
    else:
        start = time.time()
        display_graphs(sys.argv[1], sys.argv[2], sys.argv[3])
if __name__ == "__main__":
    main()
