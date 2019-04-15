"""
随机加载图片
分别从每个类别中加载12，000张图片
生成36，000大小的数据集
30000作为训练集
6000作为测试集
使用到的类别定义在constants常量中
"""
import utils.文件夹操作 as filestools
import constants
import numpy as np
import cv2
import pandas as pd

class Load_data:
    def __init__(self,datasize=5000,testsize=1000):
        self.datasize=datasize
        self.testsize = testsize
        self.dataset=self.get_all_image_info()
        self.next_index=0



    """
    加载所有的数据集信息并打乱顺序
    return [{path,name,label}]
    """
    def get_all_image_info(self):
        result=np.array([])
        for item in constants.CLASS_CATEGORY:
            image_items=filestools.get_files_of_folder(item['path'])
            print('image_items.shape',image_items.shape,'\t folder:',item['path'])
            indexs=np.random.randint(0,len(image_items),size=(self.datasize+self.testsize) if item['label']==0 else (self.datasize+self.testsize))
            image_items_result=image_items[indexs]
            for image_item in image_items_result:
                image_item['label']=item['label']

            result=np.concatenate((result,image_items_result),axis=0)
            np.random.shuffle(result)
        return result


    """
    批量加载数据集
    """
    def next_batch(self,batch_size=50):
        max=int(self.datasize/batch_size)-1
        begin=self.next_index%max*batch_size
        # print('begin:',begin,'\tend:',begin+batch_size,'\tself.dataset.shape',self.dataset.shape)
        batch_data_items=self.dataset[begin:begin+batch_size]
        self.next_index+=1
        return self.load_dataset(batch_data_items)


    """
    随机获取测试集
    """
    def get_testdata(self,size=50):
        indexs=np.random.randint(self.datasize,self.datasize+self.testsize,size=size)
        batch_data_items=self.dataset[indexs]
        return self.load_dataset(batch_data_items)


    """
    从信息中加载数据集
    """
    def load_dataset(self,batch_data_items):
        # 加载image，label
        images = []
        labels = []
        for image_item in batch_data_items:
            # print(image_item)
            image = cv2.imread(image_item['path'])
            label = np.zeros(shape=len(constants.CLASS_CATEGORY), dtype=np.float32)
            label[image_item['label']] = 1.0
            images.append(image)
            labels.append(label)

        images_result = np.array(images)
        labels_result = np.array(labels)
        return images_result, labels_result

    """
    预处理
    """
    def normalize(self,images):
        image_nor=images.astype(np.float32)/255.0
        return image_nor

if __name__=='__main__':
    load_data=Load_data()
    images_result, labels_result=load_data.next_batch(50)


