import numpy
from sklearn.linear_model import LinearRegression


def dfa(x: numpy.ndarray, max_window_size: int,  min_widow_size: int =3):
    """
    Expects X to be a time series of T rows and n columns.
    :param x:
    :param max_window_size:
    :param min_widow_size:
    :return: a list whose first element is the vector of DFA exponents for the n features and whose second element is the r_squared for the log-log regression
    """
    def unbounded(xx):
        return numpy.cumsum(xx - xx.mean(0), axis=0)

    def windows(yt, n):
        return numpy.split(yt, numpy.arange(n, yt.shape[0], step=n))

    def regression_error(window_list):
        return numpy.vstack(list(map(
            lambda x: LinearRegression().fit(numpy.arange(0, x.shape[0]).reshape(-1, 1), x).predict(
                numpy.arange(0, x.shape[0]).reshape(-1, 1)) - x, window_list)))

    def f_n(error):
        return numpy.sqrt(numpy.power(error, 2.0).sum(0) / error.shape[0])

    def f(xx, mn, mx):
        return numpy.vstack(list(map(lambda n: f_n(regression_error(windows(unbounded(xx), n))), numpy.arange(mn, mx))))

    features = numpy.log(numpy.arange(min_widow_size, max_window_size))
    y = numpy.log(f(x, mn=min_widow_size, mx=max_window_size))
    lr = LinearRegression().fit(features,y)
    r_squared = lr.score(features, y)
    hurst = lr.coef_.flatten()
    return hurst, r_squared
