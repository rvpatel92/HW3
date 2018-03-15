import numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot

def normalizeData(data):
    return data / 255.0

#https://github.com/beckernick/logistic_regression_from_scratch/blob/master/logistic_regression_scratch.ipynb
def log_likelihood(features, target, weights):
    scores = numpy.dot(features, weights)
    ll = numpy.sum( target*scores - numpy.log(1 + numpy.exp(scores)) )
    return ll

def gradDecent(features, target, weights):
    prediction = numpy.dot(features, weights)
    return numpy.sum(target * prediction - numpy.log(1 + numpy.exp(prediction)))


mnistData = numpy.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)

kf = KFold(n_splits=10)
kf.get_n_splits(mnistData)

for train_index, test_index in kf.split(mnistData):