import cv2
import tensorflow as tf
import utils.随机加载图片 as Load
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

load_data=Load.Load_data()#加载数据
tf.set_random_seed(1000)

class NetWork:

    def __init__(self,classNum=3,savepath=r'./V1/model_small.ckpt'):
        self.classNum=3
        self.savepath=savepath#训练后模型保存位置
        self.losses=[]
        self.accuracies=[]


    def buildnet(self):
        self.tf_x=tf.placeholder(tf.float32,[None,None,None,3])#28*28*3
        self.tf_y=tf.placeholder(tf.float32,[None,self.classNum])

        self.conv1=tf.layers.conv2d(self.tf_x,64,3,1,activation=tf.nn.relu6,padding='same')
        self.pool1=tf.layers.max_pooling2d(self.conv1,2,2)#14*14*32

        self.conv2 = tf.layers.conv2d(self.pool1, 128, 3, 1, activation=tf.nn.relu6, padding='same')
        self.pool2 = tf.layers.max_pooling2d(self.conv2, 2, 2)  # 7*7*128

        self.conv3=tf.layers.conv2d(self.pool2,256,7,1,activation=tf.nn.relu6)
        self.conv4=tf.layers.conv2d(self.conv3,64,1,1,activation=tf.nn.relu6)
        self.conv5=tf.layers.conv2d(self.conv4,self.classNum,1,1)

        self.output=tf.reshape(self.conv5,[-1,self.classNum])
        self.loss=tf.losses.softmax_cross_entropy(onehot_labels=self.tf_y,logits=self.output)
        # self.loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(metrics, tf.square(self.tf_y - self.output)), reduction_indices=[1]))

        self.train_op=tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.accuracy=tf.metrics.accuracy(labels=tf.argmax(self.tf_y,axis=1),predictions=tf.argmax(self.output,axis=1))[1]
        # tf.stop_gradient(self.accuracy)

        self.sess=tf.Session()
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])


    """
    训练模型
    """
    def train(self,iter=1000):
        plt.ion()
        for i in range(iter):
            bx,by=load_data.next_batch(100)
            # print(bx.shape)
            loss_,train_=self.sess.run([self.loss,self.train_op],{self.tf_x:load_data.normalize(bx),self.tf_y:by})

            if i>10 and i%100==0:
                self.losses.append(loss_)
                bxt,byt=load_data.get_testdata(100)
                acc,output_=self.sess.run([self.accuracy,self.output],{self.tf_x:load_data.normalize(bxt),self.tf_y:byt})
                print('i:',i,'\t准确率：',acc)
                print("训练集：",np.argmax(byt,axis=1))
                print(np.argmax(output_,axis=1)==np.argmax(byt,axis=1))
                df=pd.DataFrame([np.argmax(byt,axis=1),np.argmax(byt,axis=1)])
                print(df.head())
                if len(self.accuracies)>0 and acc>np.max(np.array(self.accuracies)):
                    self.save_model(self.sess)
                elif len(self.accuracies)==0:
                    self.save_model(self.sess)
                self.accuracies.append(acc)
                self.plotmodel()

        plt.ioff()
        plt.show()


    """
    保存模型
    """
    def save_model(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.savepath)

    """
    加载模型
    """
    def restore_model(self, sessin):
        saver = tf.train.Saver()
        saver.restore(sessin, self.savepath)

    def plotmodel(self):
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.accuracies)), self.losses, 'r-')
        plt.subplot(2,1,2)
        plt.plot(range(len(self.accuracies)),self.accuracies,'g-')
        plt.pause(0.1)



if __name__=='__main__':
    netWork=NetWork()
    netWork.buildnet()
    netWork.train(5000)

