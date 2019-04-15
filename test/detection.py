import model.network_large as Net
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Detection:
    def __init__(self):

        # 加载训练完成的网络
        network = Net.NetWork(savepath=r'../model/V1/model_small.ckpt')
        network.buildnet()
        network.restore_model(network.sess)

        tf.stop_gradient(network.conv5)
        self.network=network

    """
    测试
    """
    def test(self,shape=(227*6,227*6),testimage_path=r'./4.jpg'):
        self.testimage_path = testimage_path
        print('path:',self.testimage_path,'\tshape:',shape)
        image=cv2.imread(self.testimage_path)
        print('readimage:',image.shape)
        image_t_max=cv2.resize(image,shape)

        result = self.network.sess.run(self.network.conv5, {self.network.tf_x: [image_t_max]})
        print('result.shape:',result.shape)
        result_grid = result.argmax(axis=3)[0].astype(np.int)
        result_grid1 = result.max(axis=3)[0]

        plt.subplot(2,2,1)
        # plt.scatter(x, y, c=c)
        plt.imshow(result_grid)
        plt.subplot(2,2, 2)
        plt.imshow(result_grid1)
        plt.subplot(2, 2, 3)
        plt.imshow(image_t_max)
        plt.show()


        return result,result_grid,result_grid1


if __name__=='__main__':
    detection=Detection()
    result,result_grid,result_grid1=detection.test()




