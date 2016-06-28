import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import math
import scipy.stats as stats
from pylab import figure, cm
import sys
from scipy import signal
from scipy import interpolate
from scipy.optimize import leastsq
from matplotlib.colors import LogNorm


class Angle_Average(object):
    def __init__(self):
        pass


    def double_gaussian(self, x, params ):
        (c1, mu1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3) = params
        res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
              + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ) \
              + c3 * np.exp( - (x - mu3)**2.0 / (2.0 * sigma3**2.0) )
        return res

    def double_gaussian_fit(self, params):
        x = np.arange(450)
        fit = self.double_gaussian(x, params)
        return (fit - self.angle_and_intensity_average_interp)

    def get_data(file_name):
        data = np.genfromtxt(file_name, dtype=float, delimiter=None,
                             skip_header=2, names=["Qx", "Qy", "I(Qx,Qy)", "err(I)"])
        shape_x = len(np.unique(data['Qx']))
        shape_y = len(np.unique(data['Qy']))
        data_x = data['Qx']#.reshape(shape_x, shape_y)
        data_y = data['Qy']#.reshape(shape_x, shape_y)
        data_z = data['IQxQy']#.reshape(shape_x, shape_y)
        return data_x, data_y, data_z

    def func(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i+1]
            wid = params[i+2]
            y = y + amp * np.exp( -((x - ctr)/wid)**2)
        return y


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

        angle_and_intensity_average = angle_and_intensity_sum / angle_and_intensity_counts
        angle_and_intensity_average = np.nan_to_num(angle_and_intensity_average) # because division by 0
        angle_and_intensity_average = np.tile(angle_and_intensity_average, 2) # duplicates array

        return angle_and_intensity_average[:450]

    def res(p, y, x):
        m, dm, sd1, sd2,b = p
        m1 = m
        m2 = m1 + dm
        y_fit = Angle_Average.norm(x, m1, sd1, b) + Angle_Average.norm(x, m2, sd2, b)
        err = y - y_fit
        return err

    def norm(x, mean, sd, b):
        norm = []
        for i in range(x.size):
            norm += [ (1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] - mean)**2/(2*sd**2))) + b]
        return np.array(norm)

    def do_the_job(self, file_name):
        data_x, data_y, data_z = Angle_Average.get_data(file_name)
        shape_x = len(np.unique(data_x))
        shape_y = len(np.unique(data_y))
        X = data_x.reshape(shape_x, shape_y)
        Y = data_y.reshape(shape_x, shape_y)
        Z = data_z.reshape(shape_x, shape_y)

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax1.pcolormesh(X,Y,Z)

        angle_and_intensity_average = Angle_Average.radial_average(data_x, data_y, data_z)

        # normalize to 1
        angle_and_intensity_average = (angle_and_intensity_average - angle_and_intensity_average.min()) / (angle_and_intensity_average.max() - angle_and_intensity_average.min())

        x = np.arange(450)

        ax2 = fig.add_subplot(122)
        ax2.plot(x,angle_and_intensity_average,'b.')

        n_bins = 50
        self.bin_means, bin_edges, binnumber = stats.binned_statistic(x,
                                                                    angle_and_intensity_average,
                                                                    statistic='mean',
                                                                    bins=n_bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        self.bin_centers = bin_edges[1:] - bin_width/2
        # ax2.plot(self.bin_centers,self.bin_means,'r')

        # interpolate 0s
        x_new = np.arange(450)
        rbf = interpolate.Rbf(self.bin_centers, self.bin_means)
        self.angle_and_intensity_average_interp = rbf(x_new)
        ax2.plot(x_new,self.angle_and_intensity_average_interp, "black", ls ='--',label="RBF Interpolation")

        p = [1, 0, 5, 1, 180, 10, 1, 360, 10] # c1, mu1, sigma1, c2, mu2, sigma2
        lsq = leastsq(self.double_gaussian_fit, p)
        fit = self.double_gaussian(x_new, params= lsq[0])
        results = lsq[0].reshape(-1,3)
        FWHM = 2*np.sqrt(2*np.log(2))*results[1][2]
        print("******** {}".format(lsq[1]))
        print("FWHM: " + str(FWHM))

        for res in results:
            print ("amplitude, position, sigma: ", res)

        ax2.plot(x_new,fit, c = 'r', label = 'Gaussian fit')
        ax2.legend()
        plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage " + sys.argv[0]+ " Iqxy.dat file")
    else:
        angle = Angle_Average()
        angle.do_the_job(sys.argv[1])

if __name__ == "__main__":
    main()
