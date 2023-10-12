import os

from VPython.Layer import *
from VPython.Activator import *
from abc import abstractmethod
import numpy as np
import random
from datetime import datetime
from VPython.Layer import epsilon
import h5py
import pandas as pd

def listExpend(x: np.array, times: int = 0):
    x = x.tolist()
    y = x.copy()
    for i in range(1, times):
        for j in range(len(y)):
            x.append(y[j])
    for i in range(len(x)):
        x[i] = np.array(x[i])
    return x


class LOSSFunc(object):
    @abstractmethod
    def __init__(self, ):
        pass

    @abstractmethod
    def __call__(self, f, res):
        pass


class Cross_Entropy(LOSSFunc):
    def __init__(self, ):
        pass

    def __call__(self, f, res):
        f = np.array(f)
        res = np.array(res).flatten()
        index = np.argmax(res)
        derivation = np.zeros((1, res.shape[0]))
        derivation[0, index] = -1 / f[index]
        loss = -np.log(f[index])
        return derivation, loss
        pass

class Square_Difference1(LOSSFunc):
    def __init__(self, ):
        pass

    def __call__(self, f, res):
        f = np.array(f)
        res = np.array(res).flatten()
        loss =np.square(f-res)
        derivation =np.abs(np.log(np.abs(f-res).astype(np.float64)))*2*(f-res)
        return derivation, loss
        pass
class Square_Difference2(LOSSFunc):
    def __init__(self, ):
        pass

    def __call__(self, f, res):
        f = np.array(f)
        res = np.array(res).flatten()
        loss =np.square(f-res)
        derivation =2*(f-res)
        return derivation, loss
        pass
class Model(object):
    def __init__(self, input: Tensor = None, output: Tensor = None):
        self.input = input
        self.output = output
        if output is not None:
            self.layer = output.getHiddenLayer()

    def compileLoss(self, lossFunc: LOSSFunc = None):
        self.lossFunc = lossFunc
        pass

    def compileRegular(self, regularization: Regularization = None):
        self.regularization = regularization
        pass

    def compileOptimizer(self, optimizer: SGD):
        self.optimizer = optimizer
        pass

    def predict(self, input):
        input = np.array(input)
        for item in self.layer:
            input = item.forward(input)

        return input

    def outPredict(self, input):
        output = []
        for item in input:
            output.append(self.predict(item))

        return output

    def evaluate(self, xtest, ytest, filename: str = None):
        ypre = self.outPredict(xtest)
        size = len(list(ytest))
        TP = 0
        for i in range(len(ytest)):
            if np.argmax(ypre[i]) == np.argmax(ytest[i]):
                TP += 1
        if filename is not None:
            if not os.path.exists(filename):
                f = open(filename, "w")
                f.close()
            with open(filename, 'a') as f:
                f.write("accuracy:%.8f\n" % (TP / size))
            print("accuracy:%.8f\n" % (TP / size))
        return TP / size

    def fit(self, xTrain, yTrain, iteration: int = 1, log: bool = True, filename: str = None, xtest=None, ytest=None,
            step=None):
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        input = 0
        output = 0
        feed = 0
        sumloss = 0
        loss = 0
        error = False
        lossList = []
        accdict = {"iter": [], "accuracy": []}
        lr = self.optimizer.lr
        print(iteration)
        for i in range(iteration):
            testIndex = random.randint(0, len(xTrain) - 1)
            input = xTrain[testIndex]
            input = self.predict(input)
            derivation, loss = self.lossFunc(f=input, res=yTrain[testIndex])
            if np.isnan(loss[0][0]):
                error = True
                print("Error")
                break
            for item in self.layer[-1::-1]:
                derivation = item.feedBackward(derivation, self.optimizer.clipvalue)
                derivation = np.array(derivation)
            lossList.append(loss)
            lr = lr * self.optimizer.decay
            for item in self.layer:
                if error:
                    item.abort()
                    print("Error")
                    return
                else:
                    item.parameterUpdate(1, lr, self.regularization)
            sumloss = sumloss + loss[0][0]

            # if xtest is not None and ytest is not None and step is not None:
            #     if i % step == 0:

            #         acc = self.evaluate(xtest, ytest)
            #         accdict['iter'].append(i)
            #         accdict['accuracy'].append(acc)
            #         if acc > 0.92:
            #             return
            if log:
                if error:
                    if filename is None:
                        print("iteration=%d, error!" % i)
                    self.logWeights()
                else:
                    if filename is None:
                        print("iteration=%d,loss = %.8f" % (i, sumloss))
            if i % 50 == 0:
                input = xTrain[testIndex]
                input = self.predict(input)
                derivation1, loss1 = self.lossFunc(f=input, res=yTrain[testIndex])
                print("iteration=%d,loss = %.8f,loss1=%.8f,delta=%+.8f, pre=%d,test=%d" % (
                i, sumloss, loss1, loss1 - sumloss, np.argmax(input), np.argmax(yTrain[testIndex])))

            sumloss = 0

        if log:
            if filename is not None:
                if not os.path.exists(filename):
                    os.makedirs(filename)
                lossList = {'loss': lossList}
                data_pd = pd.DataFrame(lossList)
                data_pd.to_csv(filename + '/loss.csv')

                accdict = pd.DataFrame(accdict)
                accdict.to_csv(filename + "/acc.csv")

    def logModel(self, filename: str = None):
        if filename is None:
            now = datetime.now()
            ticks = now.strftime("%y_%m_%d_%H_%M_%S")
            filename = "./log/" + ticks + ".h5"
        struct_layer = ""
        for item in self.layer:
            struct_layer += (item.name + " ")
        with h5py.File(filename, "w") as f:
            subgroup = f.create_group("setting")
            subgroup.create_dataset("input", np.array(self.input.shape))
            dt = h5py.special_dtype(vlen=str)
            subgroup.attrs.create("struct", struct_layer, dtype=dt)
            dict = self.optimizer.log()
            subgroup.attrs.create('lr', dict['lr'])
            subgroup.attrs.create('decay', dict["decay"])
            subgroup.attrs.create('clipval', dict['clipval'])
            subgroup.attrs.create("model", len(self.layer))

            # subgroup.attrs.create('L2', self.regularization.lamd if self.regularization is not None else 0)
        for item in self.layer:
            item.logLayer(filename)
        return filename

    def load_model(self, filename):
        if not os.path.exists(filename):
            print("Exception:filepath error!")
            return
        with h5py.File(filename, 'r') as f:
            subgroup = f['setting']
            dict_1 = dict(subgroup.attrs.items())
            lr = dict_1['lr']
            decay = dict_1['decay']
            clip = dict_1['clipval']
            clip = clip if clip > 1e-1 else None
            opt = SGD(lr=lr, decay=decay, clipvalue=clip)
            self.compileOptimizer(opt)
            self.compileOptimizer(Cross_Entropy())
            # self.compileRegular(L2Regularization(lamd=dict_1['L2']))
            struct = dict_1['struct'].split()
            self.input = Tensor(*subgroup['input'].shape)
            out = self.input
            for i in struct:
                subgroup = f[i]
                dict_1 = dict(subgroup.attrs.items())
                if dict_1['Layer'] == 'Convolution2D':
                    act = dict_1['activator']
                    name = dict_1['name']
                    filter = dict_1['filters']
                    k_0 = dict_1['kernel_0']
                    k_1 = dict_1['kernel_1']
                    out = Convolution2D(filters=filter, kernel=(k_0, k_1), activation=act, name=name)(out)
                    layer = out.getHiddenLayer()[-1]
                    layer.loadLayer(weights=subgroup['weights'][:])
                    if dict_1['bias']:
                        layer.loadLayer(bias=subgroup['bias'][:])

                elif dict_1['Layer'] == 'Dense':
                    n = dict_1['neurons']
                    name = dict_1['name']
                    act = dict_1['activator']
                    out = Dense(neurons=n, activation=act, name=name)(out)
                    layer = out.getHiddenLayer()[-1]
                    layer.loadLayer(weights=subgroup['weights'][:])
                    if dict_1['bias']:
                        layer.loadLayer(bias=subgroup['bias'][:])
                elif dict_1['Layer'] == 'MaxPooling2D':
                    s_0 = dict_1['shape_0']
                    s_1 = dict_1['shape_1']
                    name = dict_1['name']
                    out = MaxPooling2D(shape=(s_0, s_1), name=name)(out)
                elif dict_1['Layer'] == "Flatten":
                    name = dict_1['name']
                    out = Flatten(name=name)(out)
            self.output = out
            self.layer = out.getHiddenLayer()


