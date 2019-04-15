import cv2;
import numpy as np;
import constants
import utils.文件夹操作 as filestool

class ReadVideoToImgs(object):
    def __init__(self,path=None,outpath=None,label=0):
        self.path=constants.DEFAULT_VIDEO_TEST_PATH if path is None else path
        self.outpath=constants.DEFAULT_IMAGE_ROOT_PATH if outpath is None else outpath
        self.label=label
        self.count=0 #读取到的视频帧数

    """
    读取path中的视频并保存为图片
    """
    def readvideo_saveas_imgs_with_shape(self,shape= (256, 256)):
        cap=cv2.VideoCapture(self.path)#从path中读取视频，缺省值为从摄像头中读取视频
        filestool.check_remove_folder(r'{0}/{1}'.format(self.outpath,self.label))
        filestool.check_create_folder(r'{0}/{1}'.format(self.outpath,self.label))
        while(cap.isOpened):
            ret,frame=cap.read()
            if ret==True and frame.shape[0]>200:
                #保存读取到的视频帧

                #图片缩放为224*224
                center_frame=cv2.resize(frame,shape)
                cv2.imshow('center_frame', center_frame)
                cv2.imwrite(r'{0}/{1}/{2}.jpg'.format(self.outpath,self.label,self.count),center_frame)
                self.count += 1
                print('已完成：',self.count)
                cv2.imshow('Video',frame)
                cv2.waitKey(10)
            else:
                break
        cv2.destroyAllWindows()
        cap.release()



if __name__=='__main__':
    print("开始运行")
    for i in range(constants.CLASSNUM):
        readVideoToImgs=ReadVideoToImgs(path=r'{0}/{1}.mp4'.format(constants.DEFAULT_VIDEO_PATH,i)
                                                                    ,label=i)
        readVideoToImgs.readvideo_saveas_imgs_with_shape((227,227))
        print('已完成：',i)