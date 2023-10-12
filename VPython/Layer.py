import numpy as np
import VPython.Activator as Actor
from math import ceil
from abc import abstractmethod
import h5py
import os

epsilon = 1e-8


class Layer(object):
    def __init__(self):
        self.name = 'Layer'
        pass

    @abstractmethod
    def get_layer(self):
        pass

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def feedBackward(self, output, clipval):
        pass

    @abstractmethod
    def loadLayer(self):
        pass

    @abstractmethod
    def __call__(self, data):
        pass


class Regularization(object):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data):
        pass


class L2Regularization(Regularization):
    def __init__(self, lamd: float = 0.001):
        self.lamd = lamd

    def __call__(self, data: np.array, num: int):
        def L2Regular(x):
            y = x - self.lamd * x 
            return y

        self.L2Regular = np.frompyfunc(L2Regular, 1, 1)
        res = self.L2Regular(data)
        return res


class Tensor(object):
    def __init__(self, *args):
        self.shape = args
        self.layer = []
        pass

    def __call__(self, input=None):
        return self.data

    def addLayer(self, layer: Layer, temp=None):
        if temp != None:
            a = temp.getHiddenLayer()
            if (len(a) > 0):
                for i in a:
                    self.layer.append(i)
        self.layer.append(layer)

    def getHiddenLayer(self) -> list[Layer]:
        return self.layer

    def get_layer(self):
        return self.shape


class SGD(object):
    def __init__(self, lr: float = 0.01, decay: float = 1.0, clipvalue: float = None):
        self.lr = lr
        self.decay = decay
        self.clip = clipvalue

        def clip(x):
            if np.isnan(x):
                x = np.random.rand()
            if np.isinf(x):
                x = np.random.rand()
            if clipvalue is not None:
                if x < -clipvalue:
                    x = -clipvalue
                if x > clipvalue:
                    x = clipvalue
            return x

        if clipvalue is None:
            self.clip = np.nan
        self.clipvalue = np.frompyfunc(clip, 1, 1)

    def __call__(self, *args, **kwargs):
        pass

    def log(self):
        dict_1 = {"lr": self.lr, "decay": self.decay, "clipval": self.clip if not np.isnan(self.clip) else 0}
        return dict_1


class Dense(Layer):  # epsilon
    order = 0

    def __init__(self, inputShape: tuple = None, neurons: int = None, activation: str = None,
                 biasUsed: bool = True, name: str = None):
        self.frozen = False
        self.layer = 'Dense'
        self.act_div_res = None
        self.clipval = None
        self.decay = None
        self.now = None
        self.learningRate = None
        self.loss_div_bias = None
        self.loss_div_weights = None
        self.result = None
        self.input = None
        self.backPropagationB = None
        self.lastDim = None
        self.backPropagationW = None
        self.weights = None
        if name is None:
            self.name = "Dense" + str(Dense.order)
        else:
            self.name = name
        Dense.order += 1
        self.neurons = neurons
        self.activation = activation
        self.biasUsed = biasUsed
        self.bias = np.random.rand(self.neurons, 1)
        self.activation = Actor.interceptor(activation)
        self.inputShape = inputShape
        self.inputOK()

    def inputOK(self):
        if self.inputShape is not None:
            self.lastDim = self.inputShape[0]
            self.weights = np.random.standard_normal((self.neurons, self.inputShape[0]))/ np.sqrt(self.lastDim)
            # self.weights = np.random.standard_normal((self.neurons, self.inputShape[0]))
            self.bias = np.random.standard_normal((self.neurons, 1)) / np.sqrt(self.lastDim)
            # self.bias = np.random.standard_normal((self.neurons, 1))
            self.backPropagationW = np.zeros((self.neurons, self.inputShape[0]))
            self.backPropagationB = np.zeros((self.neurons, 1))

    def __call__(self, input: Tensor = None):

        self.inputShape = input.get_layer()
        self.inputOK()
        self.output = Tensor(self.neurons)
        self.output.addLayer(self, input)
        return self.output

    def forward(self, input):
        self.input = np.array(input)
        self.input= self.input.reshape(self.lastDim,1)
        
        self.result = np.matmul(self.weights, self.input);
        if self.biasUsed:
            self.result = self.result + self.bias
        self.result = self.activation(self.result)
        self.act_div_res = self.activation.derivation(self.result, all=True)
        # print("input",self.input.shape)
        # print("result",self.result.shape)
        # print("weights",self.weights.shape)
        # print("act_div_res",self.act_div_res.shape)
        # print("")
        return self.result

    def feedBackward(self, feedback: np.array):
        # print("feedback",feedback.shape)
        # print("act_div_res",self.act_div_res.shape)

        self.loss_div_bias = np.matmul(feedback, self.act_div_res)
        self.loss_div_weights = np.matmul(self.loss_div_bias.T,self.input.T)
        self.backPropagationW = self.backPropagationW + np.array(self.loss_div_weights)
        if self.biasUsed:
            self.backPropagationB = self.backPropagationB + np.array(self.loss_div_bias.T)
        backPropagation = np.matmul(self.loss_div_bias, self.weights)
        return backPropagation

    def parameterUpdate(self, sum: int,  lr, regularization: Regularization = None,):
        if self.frozen == False:
            self.weights = self.weights - np.array(lr / sum * self.backPropagationW)
            self.bias = self.bias - np.array(lr / sum * self.backPropagationB)

        self.abort()
        pass
    def parameterDecay(self, sum: int,  lr, regularization: Regularization = None,):
        if regularization is not None & self.frozen==False:
            self.weights = regularization(data=self.weights, num=sum)
            self.bias = regularization(data=self.bias, num=sum)
        pass

    def abort(self):
        self.backPropagationW = np.zeros((self.neurons, self.inputShape[0]))
        self.backPropagationB = np.zeros((self.neurons, 1))

    def logLayer(self, filename: str = None):
        if filename is not None:
            if not os.path.exists(filename):
                f = h5py.File(filename, 'w')
                f.close()
            with h5py.File(filename, 'a') as f:
                dt = h5py.special_dtype(vlen=str)
                data = np.array([["Dense"], [self.activation.name], [self.name]])
                subgroup = f.create_group(self.name)
                subgroup.attrs.create("Layer", "Dense", dtype=dt)
                subgroup.attrs.create("activator", self.activation.name, dtype=dt)
                subgroup.attrs.create("name", self.name, dtype=dt)
                subgroup.attrs.create("bias", self.biasUsed)
                subgroup.attrs.create("neurons", self.neurons)
                subgroup.create_dataset(name="weights", data=np.array(self.weights, dtype=np.float))
                if self.biasUsed:
                    subgroup.create_dataset(name="bias", data=np.array(self.bias, dtype=np.float))

        dict_1 = {"Layer": "Dense", "neurons": self.neurons, "activator": self.activation.name, "name": self.name,
                  "weights": self.weights}
        if self.biasUsed:
            dict_2 = {"bias": self.bias}
            dict_1.update(dict_2)

        return dict_1
    def load(self,weights,bias):
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def loadLayer(self, filename=None, weights=None, bias=None):
        if filename is None:
            if weights is not None:
                weights = np.array(weights)
                assert self.weights.shape == weights.shape
                self.weights = weights
            if bias is not None:
                bias = np.array(bias)
                assert self.bias.shape == bias.shape
                self.bias = bias
        pass

    def __repr__(self):
        return self.name


class Convolution2D(Layer):
    order = 0

    def __init__(self, inputShape: tuple = None, filters: int = None, kernel: tuple = None, activation: str = None,
                 biasUsed: bool = False, name: str = None):
        self.layer = 'Convolution2D'
        self.frozen == False
        self.loss_div_bias = None
        self.act_div_res = None
        self.loss_div_weights = None
        self.result = None
        self.input = None
        self.backPropagationB = None
        self.backPropagationW = None
        self.weights = None
        self.bias = None
        self.outShape = None
        if name is None:
            self.name = "Convolution2D" + str(Convolution2D.order)
        else:
            self.name = name
        Convolution2D.order += 1
        self.filters = filters
        self.kernel = kernel
        self.biasUsed = biasUsed
        self.activation = Actor.interceptor(activation)
        self.inputShape = inputShape
        self.clipval = None
        self.inputOK()

    def inputOK(self):
        if self.inputShape is not None:
            if len(self.inputShape) == 2:
                self.inputShape = (1, self.inputShape[0], self.inputShape[1])
            temp = np.sqrt(3 * self.inputShape[0] * self.inputShape[1] * self.inputShape[0]/self.filters)
            inputShape = self.inputShape
            self.outShape = (self.filters, self.inputShape[1] - self.kernel[0] + 1, self.inputShape[2] - self.kernel[1] + 1)
            self.bias = np.random.standard_normal(self.outShape)/temp
            self.weights = np.random.standard_normal((self.filters, inputShape[0], self.kernel[0], self.kernel[1]))/temp
            self.backPropagationW = np.zeros((self.filters, inputShape[0], self.kernel[0], self.kernel[1]))
            self.backPropagationB = np.zeros(self.outShape)
            self.backPropagation = np.zeros(self.inputShape)

    def __call__(self, input: Tensor = None):
        self.inputShape = input.get_layer()
        self.inputOK()
        self.output = Tensor(*self.outShape)
        self.output.addLayer(self, input)
        return self.output

    def forward(self, input):
        self.input = np.array(input)
        if len(self.input.shape) == 2:
            self.input = self.input.reshape(1, self.input.shape[0], self.input.shape[1])

        self.result = np.zeros(self.outShape)
        for i in range(self.outShape[0]):
            for j in range(self.outShape[1]):
                for k in range(self.outShape[2]):
                    k_h = self.kernel[0]
                    k_w = self.kernel[1]
                    self.result[i, j, k] = np.sum(self.input[:, j:j + k_h, k:k + k_w] * self.weights[i, :, :, :])

        if self.biasUsed:
            self.result = self.result + self.bias
        self.result = self.activation(self.result)
        self.act_div_res = self.activation.derivation(self.result, all=False)
        self.loss_div_weights = np.zeros_like(self.weights)
        return self.result

    def feedBackward(self, feedback: np.array):
        self.loss_div_bias = np.array(feedback * self.act_div_res)
        # 反卷积，loss_div_bias为卷积核，给输入做卷积
        for i in range(self.filters):
            for j in range(self.inputShape[0]):
                for l in range(self.kernel[0]):
                    for m in range(self.kernel[1]):
                        k_h = self.outShape[1]
                        k_w = self.outShape[2]
                        self.loss_div_weights[i, j, l, m] = np.sum(
                            self.loss_div_bias[i, :, :] * self.input[j, l:l + k_h, m:m + k_w])

        self.backPropagationW = self.backPropagationW + self.loss_div_weights
        self.backPropagationB = self.backPropagationB + self.loss_div_bias

        backPropagation = np.zeros(self.inputShape)
        k_180 = np.rot90(self.weights, 2, (2, 3))
        pad1 = self.kernel[0] - 1
        pad2 = self.kernel[1] - 1
        temp = np.pad(self.loss_div_bias, ((0, 0), (pad1, pad1), (pad2, pad2)), 'constant', constant_values=(0, 0))
        for f in range(self.filters):
            for i in range(self.inputShape[0]):
                for j in range(self.inputShape[1]):
                    for k in range(self.inputShape[2]):
                        k_h = self.kernel[0]
                        k_w = self.kernel[1]
                        backPropagation[i, j, k] += np.sum(np.array(k_180[f, i, :, :]) * np.array(temp[f, j:j+k_h, k:k+k_w]))
        return backPropagation

    def parameterUpdate(self, sum: int, lr, regularization: Regularization = None):
        if self.frozen == False:
            self.weights = self.weights - lr / sum * self.backPropagationW
            if self.biasUsed:
                self.bias = self.bias - lr / sum * self.backPropagationB

        self.abort()
    def parameterDecay(self, sum: int,  lr, regularization: Regularization = None,):
        if regularization is not None & self.frozen == False:
            self.weights = regularization(data=self.weights, num=sum)
            self.bias = regularization(data=self.bias, num=sum)
        pass
    def abort(self):
        self.backPropagationW = np.zeros((self.filters, self.inputShape[0], self.kernel[0], self.kernel[1]))
        self.backPropagationB = np.zeros(self.outShape)
        self.backPropagation = np.zeros(self.inputShape)

    def logLayer(self, filename: str = None):
        if filename is not None:
            if not os.path.exists(filename):
                f = h5py.File(filename, 'w')
                f.close()
            with h5py.File(filename, 'a') as f:
                dt = h5py.special_dtype(vlen=str)
                subgroup = f.create_group(self.name)
                subgroup.attrs.create("Layer", "Convolution2D", dtype=dt)
                subgroup.attrs.create("activator", self.activation.name, dtype=dt)
                subgroup.attrs.create("name", self.name, dtype=dt)
                subgroup.attrs.create("filters", self.filters)
                subgroup.attrs.create("kernel_0", self.kernel[0])
                subgroup.attrs.create("kernel_1", self.kernel[1])
                subgroup.attrs.create("bias", self.biasUsed)


                subgroup.create_dataset(name="weights", data=np.array(self.weights, dtype=np.float))
                if self.biasUsed:
                    subgroup.create_dataset(name="bias", data=np.array(self.bias, dtype=np.float))
        dict_1 = {"Layer": "Convolution2D", "filters": self.filters, "kernel": self.kernel, "activator":
            self.activation.name, "name": self.name, "weights": self.weights}
        if self.biasUsed:
            dict_2 = {self.name + "_bias": self.bias}
            dict_1.update(dict_2)
        return dict_1

    def loadLayer(self, filename=None, weights=None, bias=None):
        if filename is None:
            if weights is not None:
                weights = np.array(weights)
                assert self.weights.shape == weights.shape
                self.weights = weights
            if bias is not None:
                bias = np.array(bias)
                assert self.bias.shape == bias.shape
                self.bias = bias
        pass

    def __repr__(self):
        return self.name


class MaxPooling2D(Layer):
    order = 0

    def __init__(self, shape: tuple, input: Tensor = None, name=None):
        self.layer = 'MaxPooling2D'
        self.feedback = None
        if name is not None:
            self.name = name
        else:
            self.name = "MaxPooling2D" + str(MaxPooling2D.order)
        MaxPooling2D.order = MaxPooling2D.order + 1
        self.inputShape = input
        self.shape = shape

    def __call__(self, input: Tensor = None):
        self.inputShape = input.get_layer()
        self.outputShape = (
        self.inputShape[0], ceil(self.inputShape[1] / self.shape[0]), ceil(self.inputShape[2] / self.shape[1]))
        self.output = Tensor(*self.outputShape)
        self.output.addLayer(self, input)
        return self.output

    def forward(self, input):
        input = np.array(input)
        self.res = np.zeros(self.outputShape)
        self.feedback = np.zeros(input.shape)
        for i in range(self.outputShape[0]):
            for j in range(self.outputShape[1]):
                for k in range(self.outputShape[2]):
                    temp = input[i, j * self.shape[0]:min((j + 1) * self.shape[0], self.inputShape[1]),
                           k * self.shape[1]:min((k + 1) * self.shape[1], self.inputShape[2])]
                    index = np.unravel_index(temp.argmax(), temp.shape)
                    self.res[i][j][k] = max(temp.flatten())
                    self.feedback[i][j * self.shape[0] + index[0]][k * self.shape[1] + index[1]] = 1
                    pass
        return self.res

    def feedBackward(self, feedback: np.array):
        self.feedback = np.repeat(np.repeat(feedback, self.shape[0], axis=1), self.shape[1], axis=2)[:,
                        :self.inputShape[1], :self.inputShape[2]] * self.feedback

        return self.feedback

    def parameterUpdate(self, sum: int, lr, regularization: Regularization = None):
        pass
    def parameterDecay(self, sum: int,  lr, regularization: Regularization = None,):
        pass
    def abort(self):
        pass

    def logLayer(self, filename: str = None):
        if filename is not None:
            if not os.path.exists(filename):
                f = h5py.File(filename, 'w')
                f.close()
            with h5py.File(filename, 'a') as f:
                dt = h5py.special_dtype(vlen=str)
                subgroup = f.create_group(self.name)
                subgroup.attrs.create("shape_0", self.shape[0])
                subgroup.attrs.create("shape_1", self.shape[1])
                subgroup.attrs.create("Layer", "MaxPooling2D", dtype=dt)
                subgroup.attrs.create("name", self.name, dtype=dt)

        dict_1 = {"Layer": "MaxPooling2D", "shape": self.shape, "name": self.name}
        return dict_1

    def loadLayer(self, filename=None, weights=None, bias=None):
        if filename is not None:
            pass
        pass

    def __repr__(self):
        return self.name


class Flatten(Layer):
    order = 0

    def __init__(self, inputShape: tuple = None, name: str = None):
        self.layer = 'Flatten'
        if name is not None:
            self.name = name
        else:
            self.name = "Flatten" + str(Flatten.order)
        Flatten.order = Flatten.order + 1
        self.inputShape = inputShape
        self.inputOK()

    def inputOK(self):
        if self.inputShape != None:
            self.temp = np.array(self.inputShape)
            x = np.cumprod(self.temp)[-1]
            self.outShape = (x, 1)

    def __call__(self, input: Tensor):
        self.inputShape = tuple(input.get_layer())
        self.inputOK()
        self.res = Tensor(*self.outShape)
        self.res.addLayer(self, input)
        return self.res

    def forward(self, input: np.array):
        self.result = np.array(input)
        self.result = self.result.flatten()
        self.result = self.result.reshape(self.result.shape[0], 1)
        return self.result

    def feedBackward(self, feedback: np.array):
        shape = self.inputShape
        return feedback.reshape(*shape)

    def parameterUpdate(self, sum: int, lr, regularization: Regularization = None):
        pass
    def parameterDecay(self, sum: int,  lr, regularization: Regularization = None,):
        pass

    def abort(self):
        pass

    def logLayer(self, filename: str = None):
        if filename is not None:
            if not os.path.exists(filename):
                f = h5py.File(filename, 'w')
                f.close()
            with h5py.File(filename, 'a') as f:
                dt = h5py.special_dtype(vlen=str)
                subgroup = f.create_group(self.name)
                subgroup.attrs.create("Layer", "Flatten", dtype=dt)
                subgroup.attrs.create("name", self.name, dtype=dt)

        dict_1 = {"Layer": "Flatten", "name": self.name}
        return dict_1

    def loadLayer(self, filename=None, weights=None, bias=None):
        if filename is not None:
            pass
        pass

    def __repr__(self):
        return self.name
