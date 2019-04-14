import cv2
import numpy as np
import constants
import utils.文件夹操作 as filestool

"""
图像增强
"""

class ImagePlus:
    def __init__(self,path=None,outpath=None):
        self.path=constants.DEFAULT_IMAGE_TEST_PATH if path is None else path
        self.outpath=self.path if outpath is None else outpath #处理后输出文件夹
        self.origin_images=filestool.get_files_of_folder(self.path)#所有原图


    """
        图像镜像
    """
    def mirro_all_images(self):

        for image_item in self.origin_images:
            image=cv2.imread(image_item['path'])
            dst=np.zeros([image.shape[1],image.shape[0],image.shape[2]],dtype=image.dtype)
            for h in range(image.shape[0]):
                for w in range(image.shape[1]):
                    dst[h,w,:]=image[w,h,:]

            dst_resize=cv2.resize(dst,dsize=(image.shape[0],image.shape[1]))
            cv2.imwrite(self.get_imagename_by_op(image_item,'mirror'),dst_resize)
        print("镜像操作完成",self.outpath)

    """
    加入椒盐噪声的函数
    """
    def SaltAndPepper(self, src, percetage=0.1):
        SP_NoiseImg = src
        SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
        for i in range(SP_NoiseNum):
            randX = np.random.randint(0, src.shape[0] - 1)
            randY = np.random.randint(0, src.shape[1] - 1)
            if np.random.randint(0, 1) == 0:
                SP_NoiseImg[randX, randY] = 0
            else:
                SP_NoiseImg[randX, randY] = 255
        return SP_NoiseImg


    def add_saltAndPepper(self):
        for image_item in self.origin_images:
            image=cv2.imread(image_item['path'])
            image_salt=self.SaltAndPepper(image)
            cv2.imwrite(self.get_imagename_by_op(image_item,'salt'),image_salt)

        print("加入椒盐噪声完成",self.outpath)


    """
    改变亮度和对比度
    """
    def Contrast_and_Brightness(self,image,alpha,beta):
        blank=np.zeros_like(image,dtype=image.dtype)
        dst = cv2.addWeighted(image, alpha, blank, 1 - alpha, beta)
        return dst

    def Contrast_and_Brightness_all(self):
        for image_item in self.origin_images:
            a,b=np.random.uniform(0.8,1.2,size=1)[0],np.random.uniform(-0.2,0.2,size=1)[0]
            #a>1,对比度增强，0-1，对比度减弱
            #b>0亮度增强，b<0 亮度降低

            image=cv2.imread(image_item['path'])
            dst=self.Contrast_and_Brightness(image,a,b)
            cv2.imwrite(self.get_imagename_by_op(image_item,'CB'),dst)
        print('改变对比度、亮度完成',self.outpath)

    """
    构造文件名
    """
    def get_imagename_by_op(self,image_item,op):
        return r'{0}/{1}.jpg'.format(self.outpath,op+"_"+image_item['name'])


if __name__=='__main__':
    for i in range(constants.CLASSNUM):
        path=r"{0}/{1}".format(constants.DEFAULT_IMAGE_ROOT_PATH,i)
        imagePlus=ImagePlus(path)
        imagePlus.Contrast_and_Brightness_all()
        imagePlus.add_saltAndPepper()
        imagePlus.mirro_all_images()

        print("imagePlus 完成：",i)


