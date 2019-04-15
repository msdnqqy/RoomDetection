import model.network_small as Net
import tensorflow as tf
import numpy as np
import cv2

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
    def test(self,shape=(1280,1280),testimage_path=r'./45.jpg'):
        self.testimage_path = testimage_path
        print('path:',self.testimage_path,'\tshape:',shape)
        image=cv2.imread(self.testimage_path)
        print('readimage:',image.shape)
        image_t_max=cv2.resize(image,shape)

        result = self.network.sess.run(self.network.conv5, {self.network.tf_x: [image_t_max]})
        print('result.shape:',result.shape)
        result_grid = result.argmax(axis=3)[0].astype(np.int)
        result_grid1 = result.max(axis=3)[0]

        x = []
        y = []
        c = []
        d = []
        for i in range(result_grid.shape[0]):
            for j in range(result_grid.shape[1]):
                x.append(i)
                y.append(j)

                if result_grid[i, j] == 0:
                    c.append('green')
                elif result_grid[i, j] == 1:
                    c.append('yellow')
                elif result_grid[i, j] == 2:
                    c.append('red')

                d.append(result_grid1[i, j])
                # c.append(result_grid[i, j])

        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x, y, c=c)
        plt.show()

        return result,result_grid,result_grid1


if __name__=='__main__':
    detection=Detection()
    result,result_grid,result_grid1=detection.test()




