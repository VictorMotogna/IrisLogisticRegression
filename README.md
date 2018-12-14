# Iris Dataset Logistic Regression

## Dataset

  - Iris Dataset - https://archive.ics.uci.edu/ml/datasets/iris
  - 3 classes x 50 instances each
  - labeled by: sepal length, sepal width, petal length, petal width
  
  ![alt text](https://i.imgur.com/oRACM5G.png "Iris Dataset - sepal length/width")
  
### Data set usages in this implementation

  - using all 4 characteristics: 
    - sepal length, sepal width, petal length, petal width
    - `X = iris.data[:, :4]`
  - using only the first characteristics
    - sepal length, sepal width
    - `X = iris.data[:, :2]`
  - using only `setosa` & `virginica`
    - for __Logistic Regression__ we need to consider binary results
    - `setosa = 0`, `virginica = 1`

![alt text](https://i.imgur.com/4kqr17x.png "Logistic Regression Dataset - sepal length/width")

## Implementations

  1. Logistic Regression from scratch
      - use a `sigmoid` function to output a result between 0 & 1
        - `return 1 / (1 + np.exp(-z))`
      - use a loss function with parameters (weights - theta) to compute the best value for them
        - initially pick random values
        - `return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()`
      - gradient descent
        - `gradient = np.dot(X.T, (h - y)) / y.shape[0]`
      - predictions
        - `def predict_probs(X, theta): return sigmoid(np.dot(X, theta))`
  2. Logistic Regression from scikit-learn
      - docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
      - __much faster__ than the scratch implementation

## Necessary tools

  - `python 3.x`
  - `pip3`
  - `matplotlib (pyplot)`
  - `sklearn`

## References
  - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
