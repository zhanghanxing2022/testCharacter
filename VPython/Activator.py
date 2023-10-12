from abc import abstractmethod
import numpy as np


def to_categorical(x, classes):
    size = len(x)
    zeros = np.zeros(classes)
    zeros = zeros.tolist()
    y = []
    for i in range(size):
        temp = zeros.copy()
        temp[x[i]] = 1
        y.append(temp)
    y = np.array(y)
    return y


def categorical_back(y):
    res = []
    for item in y:
        temp = np.array(item)
        res.append(temp.argmax())
    return res


class Activator(object):
    def __init__(self, ):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivation(self,all=True):
        pass


class Logistic(Activator):
    def __init__(self):
        self.name = "logistic"

        def logistic(x):
            return 1 / (np.exp(-x) + 1)
        def back(x):
            return (1-x)*x

        self.func = np.frompyfunc(logistic, 1, 1)
        self.back = np.frompyfunc(back, 1, 1)

    def __call__(self, x):
        self.x = np.array(x)
        self.y = self.func(self.x)
        return self.y

    def __repr__(self):
        return "logistic"

    def derivation(self, res, all=True):
        if all:
            self.res = self.back(res)
            self.res = np.diag(self.res.flatten())
        else:
            self.res = (res) * (1 - res)
        return self.res


class ReLu(Activator):
    def __init__(self, gamma=0.001):
        self.gamma = gamma
        self.name = "relu"
        def func(x):
            return x if x > 0 else self.gamma * x

        def div(x):
            return 1 if x > 0 else self.gamma

        self.func = func
        self.func = np.frompyfunc(func, 1, 1)
        self.div = np.frompyfunc(div, 1, 1)
        pass

    def __call__(self, x):
        return self.func(x)

    def __repr__(self):
        return "relu"

    def derivation(self, y, all=True):
        if all:
            y = y.reshape(y.shape[0])
            self.res = np.diag(self.div(y))
        else:
            self.res = self.div(y)
        return self.res

class X(Activator):
    def __init__(self, ):
        self.name = "x"
        def func(x):
            return x 

        def div(x):
            return 1 

        self.func = func
        self.func = np.frompyfunc(func, 1, 1)
        self.div = np.frompyfunc(div, 1, 1)
        pass

    def __call__(self, x):
        return np.array(x)

    def __repr__(self):
        return "x"

    def derivation(self, y, all=True):
        if all:
            y = y.reshape(y.shape[0])
            self.res = np.diag(self.div(y))
        else:
            self.res = self.div(y)
        return self.res

class Softmax(Activator):
    def __init__(self):
        self.name = "softmax"
        pass

    def __call__(self, x):
        x = np.array(x)

        self.x = np.array(x).flatten()
        self.x = list(self.x)
        temp = sum(self.x) / len(self.x)
        # print("0:,",self.x)
        for i in range(len(self.x)):
            self.x[i] = self.x[i] - temp
        # print("1:,", self.x)
        self.temp = []
        for i in self.x:
            try:
                self.temp.append(np.exp(i))
            except np.error_message:
                print("error!", self.x)
                pass
        self.sum = sum(self.temp)
        self.y = []
        for i in self.temp:
            self.y.append(i / self.sum)
        self.y = np.array(self.y)
        return self.y

    def __repr__(self):
        return "softmax"

    def derivation(self, res, all=None):
        self.res = np.zeros((len(self.x), len(self.x)))
        size = len(self.x)
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.res[i, i] = self.temp[i] / self.sum * (self.sum - self.temp[i]) / self.sum
                else:
                    self.res[i, j] = -self.temp[j] / self.sum * self.temp[i] / self.sum
        return self.res


def interceptor(input: str = "logistic") -> Activator:
    activator = Activator()
    if input == "logistic":
        activator = Logistic()
    elif input == "relu":
        activator = ReLu()
    elif input == "softmax":
        activator = Softmax()
    elif input == "x":
        activator = X()
    return activator
# x = [1,2,3,4,5]
# activator = Softmax()
# res = activator(x)
# print("res:", res)
# print("activator:",activator.derivation())
