import numpy as np

class KalmanFilter(object):

    def __init__(self, mean, sigma):
        self._mean = float(mean)
        self._sigma = float(sigma)

    def measurement(self, mean, sigma):
        self._mean = (self._mean * sigma + mean * self._sigma) / (sigma + self._sigma)
        self._sigma = 1. / ((1. / sigma) + (1. / self._sigma))

    def prediction(self, motion_mean, motion_sigma):
        self._mean += motion_mean
        self._sigma += motion_sigma

    def __str__(self):
        return ("Mean: {}, Var: {}".format(self._mean, self._sigma))


measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurements_sigma = 4.
motion_sigma = 2.
kalman = KalmanFilter(mean=0, sigma=0.000001)
print(kalman)
for idx, measurement in enumerate(measurements):
    kalman.measurement(mean=measurement, sigma=measurements_sigma)
    print(kalman)
    kalman.prediction(motion_mean=motion[idx], motion_sigma=motion_sigma)
    print(kalman)
