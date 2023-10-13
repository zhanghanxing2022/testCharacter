from VPython.NetWork import *
from PIL import Image
from alive_progress import alive_bar
import time
from tqdm import trange
import matplotlib.pyplot as plt

Input = Tensor(28*28,1)
out = Dense(neurons=100, activation='logistic', biasUsed=True)(Input)
out = Dense(neurons=100, activation='logistic')(out)
softmaxOut = Dense(neurons=12, activation="softmax")(out)
optimizer = SGD(lr=0.1, decay=1.0, clipvalue=10)

model = Model(input=Input, output=softmaxOut)
model.compileLoss(Cross_Entropy())
model.compileRegular(L2Regularization(0.05))
model.compileOptimizer(optimizer)


class CustomImageDataset:
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        img1_path=os.path.join(self.img_dir,'1')
        self.n = len(os.listdir(img1_path))

    def __len__(self):
        return len(self.img_labels)*self.n

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,str(int(idx/self.n )+ 1)+'/'+str(idx%self.n+1)+".bmp")
        with Image.open(img_path) as im:
            image = im.getdata()
        label = int(idx/self.n )+ 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
def transform(x):
    x = np.array(x).reshape(28*28,1)
    return x/255
def target_transform(x):
    res = np.zeros((12,1),dtype=np.float16)
    res[int(x-1)][0] = 1.0
    return res

annotations='博学笃志切问近思自由无用'
annotations=list(annotations)
print(annotations)
imageSet = CustomImageDataset(annotations,'./train/',transform,target_transform)
image, label=imageSet[2212]
image

def fit(model:Model,Train,Test,logfile, epoch:int =1,batch_size: int = 1, log: bool = True):
    nn=len(Train)
    input = 0
    sumloss = 0
    loss = 0
    error = False
    lr = model.optimizer.lr
    iteration = int(nn/batch_size)
    print("iteration",iteration)
    lastLoss = 0
    allLoss = 0.0
    worse = 0
    TrainAcc = 0
    TestAcc = 0
    LossList = []
    TrainAccList = []
    TestAccList=[]
    with alive_bar(epoch*iteration*batch_size) as bar:
        for i in range(epoch):
            TrainAcc = 0
            TestAcc = 0
            print(i)
            testIndex = list(range(nn))
            np.random.shuffle(testIndex)
            lastLoss = allLoss
            allLoss = 0.0
            for layer in model.layer:
                layer.parameterDecay(0,0,model.regularization)#l2正则化
            for j in range(iteration):

                
                sumloss=0.0
                for k in range(batch_size):
                    testIndex0 = testIndex[j*batch_size+k]
                    image,label = Train[testIndex0]
                    input = model.predict(image)
                    derivation, loss = model.lossFunc(f=input, res=label)
                    if np.isnan(loss):
                        error = True
                        print("Error")
                        break
                    for item in model.layer[-1::-1]:
                        derivation = item.feedBackward(derivation)
                        derivation = np.array(derivation)
                    if np.argmax(input) == np.argmax(label):
                        TrainAcc = TrainAcc + 1
                    
                    sumloss = sumloss + loss
                    bar()

                allLoss = allLoss + sumloss/batch_size
                for item in model.layer:
                    if error:
                        item.abort()
                        print("Error")
                        return
                    else:
                        item.parameterUpdate(batch_size, lr, model.regularization)
                # if xtest is not None and ytest is not None and step is not None:
            for j in range(len(Test)):
                image,label = Test[j]
                input = model.predict(image)
                if np.argmax(input) == np.argmax(label):
                    TestAcc = TestAcc +1
            if allLoss > lastLoss:
                worse = worse+1
                
            if worse > 3:
                lr = max(lr*0.9,0.001)
                worse  = 0
            with open(logfile,'a') as f:
                f.write("epoch=%d,loss = %.8f,trainAcc=%.8f,testAcc = %.8f\n" % (i, allLoss/iteration,TrainAcc/nn,TestAcc/len(Test)))
            if log:
                print("epoch=%d,loss = %.8f,trainAcc=%.8f,testAcc = %.8f" % (i, allLoss/iteration,TrainAcc/nn,TestAcc/len(Test)))
            lr = max(lr * model.optimizer.decay,0.001)
            LossList.append(allLoss/iteration)
            TrainAccList.append(TrainAcc/nn)
            TestAccList.append(TestAcc/len(Test))
            model.logModel("./model/character_epoch_"+str(i)+".hdf5")
    print(LossList)
    plt.plot(list(range(len(LossList))),LossList)
    plt.show()
    plt.plot(list(range(len(TrainAccList))),TrainAccList,'-g')
    plt.plot(list(range(len(TestAccList))),TestAccList,'-b')
    plt.show()
mylist = list(range(620))
rate = 0.1
logfile = "dnn1.txt"
with open(logfile, "w") as file:
    file.write("begin\n")
with open(logfile, "a") as file:
    file.write("begin\n")
Train = []
Test = []
for i in range(12):
    np.random.shuffle(mylist)
    for j in range(620):
        if j < 620*rate:
            Test.append(imageSet[i*620+mylist[j]])
        else:
            Train.append(imageSet[i*620+mylist[j]])

print(len(Train))
print(len(Test))

fit(model,Train,Test,logfile,100,24)

model.logModel("./model/characterdnn.hdf5")
def evaluate(model:Model,Train):
    nn=len(Train)
    acc = 0
    for i in range(nn):
        image, label = Train[i]
        input = model.predict(image)
        if np.argmax(input) == np.argmax(label):
            acc += 1
    acc = acc/nn
    print("acc",acc)
    
evaluate(model,Test)

