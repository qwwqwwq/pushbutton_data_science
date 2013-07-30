import argparse
import sklearn
from sk_learn_framework import TrainTestDataset
from pybrain_sklearn import PybrainNN
from pybrain.supervised.trainers import RPropMinusTrainer

### process command line args
parser = argparse.ArgumentParser(description='Use sklearn and pandas to perform supervised learning.')
parser.add_argument('--data', dest='trainfile', type=str,
                    help="Input training data.")
parser.add_argument('--save', dest='savefile', type=str,
                    help="Where model will be saved.")
parser.add_argument('--sample', type=float, dest='sample', default=0.75,
                    help="Proportion of data to train on.")
parser.add_argument('--predict', type=str, dest='predictfile', default=None,
                    help="Input testing data")
parser.add_argument('--log', type=str, dest='logfile', default='sklearn.log',
                    help="Log destination")
parser.add_argument('--k', type=int, dest='k', default='5',
                    help="Number of features to use.")
args = parser.parse_args()


if __name__ == '__main__':
    d = TrainTestDataset(args.trainfile, args.predictfile, Kfeats = args.k)
    d.add_model(PybrainNN())
    d.add_model(sklearn.linear_model.LinearRegression())
    d.add_model(sklearn.svm.SVR())
    d.add_model(sklearn.linear_model.BayesianRidge())
    if args.predictfile:
	d.predict_best()
    else:
	d.cv()


