import cv2
import tensorflow as tf
import utils.随机加载图片 as Load

load_data=Load.Load_data()#加载数据

class NetWork:

    def __init__(self,savepath):
        self.savepath=savepath#训练后模型保存位置

    def buildnet(self):
