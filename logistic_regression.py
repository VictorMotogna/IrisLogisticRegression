import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import datetime
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

def plotDataset():
    # The indices of the features that we are plotting
    x_index = 0
    y_index = 1

    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.figure(figsize=(10, 8))
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

    plt.tight_layout()
    plt.show()

X = iris.data[:, :2]
y = (iris.target != 0) * 1

# figure with 2 types of iris: setosa & virginica
def plotRestrictedDataset():
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    plt.show()

def sklearnLogisticRegression(iterations):
    model = LogisticRegression(C=iterations, solver='lbfgs')
    # solver used to avoid FutureWarning on default solver
    model.fit(X, y)
    preds = model.predict(X)
    (preds == y).mean()
    print("sklearn accuracy: ", model.score(X, y))


def scratchLogisticRegression(iterations):
    model = LogisticRegressionScratch(lr=0.1, num_iter=iterations)
    model.fit(X, y)
    preds = model.predict(X)
    print("scratch accuracy: ", (preds == y).mean())

    
def plotScratchLogisticRegression():
    model = LogisticRegressionScratch(lr=0.1, num_iter=50000)
    model.fit(X, y)

    preds = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predict_prob(grid).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')

    plt.show()

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, num_iter=10000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()

# tests: 10.000, 50.000, 200.000, 300.000
numberOfIterations = 300000

scratchMethodStart = datetime.datetime.now()
scratchLogisticRegression(numberOfIterations)
scratchMethodEnd = datetime.datetime.now()
print("scratch duration: ", scratchMethodEnd-scratchMethodStart)

skMethodStart = datetime.datetime.now()
sklearnLogisticRegression(numberOfIterations)
skMethodEnd = datetime.datetime.now()
print("sklearn duration: ", skMethodEnd-skMethodStart)

# plotDataset()
# plotScratchLogisticRegression()