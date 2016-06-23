import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import math
import scipy.stats as stats
from pylab import figure, cm

def get_data(file_name):
    data = np.genfromtxt(file_name, dtype=float, delimiter=None,
                         skip_header=2, names=["Qx", "Qy", "I(Qx,Qy)", "err(I)"])
    shape_x = len(np.unique(data['Qx']))
    shape_y = len(np.unique(data['Qy']))
    data_x = data['Qx'].reshape(shape_x, shape_y)
    data_y = data['Qy'].reshape(shape_x, shape_y)
    data_z = data['IQxQy'].reshape(shape_x, shape_y)
    return data_x, data_y, data_z


def main():
    data_x, data_y, data_z = get_data("EQ-SANS/anisotropic_with_peak/dPF-0RH-hiq_Iqxy.dat")
    theta = np.linspace(0, 360, 360)
    sums = []
    for angle in range(len(theta)):
        x0, y0 = 0, 0 # These are in _pixel_ coordinates!!
        x1 = 1
        y1 = math.tan(angle * (180/math.pi))
        num = 1000
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        zi = scipy.ndimage.map_coordinates(data_z, np.vstack((x,y)))
        sums.append(np.average(zi))
    sums = np.absolute(sums)
    max1 = np.amax(sums)
    print(max1)
    max_array = []
    for number in range(0, len(sums)):
        if sums[number] == max1:
            max_array.append(number)
    # shape, loc, scale = stats.lognorm.fit(sums)
    # pdf_fitted = stats.lognorm.pdf(sums, shape, loc=loc, scale=scale)
    # plt.plot(theta, pdf_fitted)
    # plt.show()
    plt.imshow(data_z)
    # plt.plot(theta, sums)
    plt.show()

if __name__ == "__main__":
    main()
