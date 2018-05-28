#-*-coding:utf8-*-

from dataSet import DataSet
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout
import numpy as np

#建立一个基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "./model/model.h5"  # 模型进行存储和读取的地方，hdf5是一种针对大量数据进行组织和存储的文件格式
    IMAGE_SIZE = 128  # 模型接受的人脸图片一定得是128*128的
    def __init__(self):
        self.model = None
    #读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self,dataset):
        self.dataset = dataset

    #建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential() #序贯模型是多个网络层的线性堆叠
        self.model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same', #自动补零，使得输出尺寸在过滤窗口步幅为1的情况下与输入尺寸相同
                data_format='channels_last',#“channels_last”对应原本的“tf”
                input_shape=self.dataset.X_train.shape[1:]
            )
        )
        self.model.add(Activation('relu'))#relu函数 类似线性 收敛速度快
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # 全连接层
        self.model.add(Flatten())  # 先将前一层输出的二维特征图flatten为一维的
        self.model.add(Dense(512)) #隐藏层
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    #进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='adam',  #有很多可选的optimizer（优化器），Adam梯度下降
            loss='categorical_crossentropy',  #多类的对数损失,你可以选用squared_hinge作为loss看看哪个好
            metrics=['accuracy']
        )
        #epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本,verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        self.model.fit(self.dataset.X_train,
                       self.dataset.Y_train,
                       batch_size=20,
                       epochs=10,
                       verbose=2
                       )
    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    #需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self,img):
        img = img.reshape((1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))
        img = img.astype('float32')
        img = img/255.0
        result = self.model.predict_proba(img)  #测算一下该img属于某个label的概率
        max_index = np.argmax(result) #找出概率最高的
        return max_index,result[0][max_index] #第一个参数为概率最高的label的index,第二个参数为对应概率

if __name__ == '__main__':
    dataset = DataSet('./image/trainfaces')
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()














