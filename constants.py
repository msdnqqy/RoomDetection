"""
常量
"""
import os
print("root:",os.getcwd())

ROOT_PATH=r'C:/Users/Administrator/Desktop/RoomDetection/'
#视频存储地址
DEFAULT_VIDEO_PATH=ROOT_PATH+r'dataset/video'

#图片存储根地址
DEFAULT_IMAGE_ROOT_PATH=ROOT_PATH+r'dataset/images'

#测试视频地址
DEFAULT_VIDEO_TEST_PATH=ROOT_PATH+r'dataset/video/1.mp4'

#测试图像地址
DEFAULT_IMAGE_TEST_PATH=ROOT_PATH+r'dataset/images/0'

CLASSNUM=3

CLASS_CATEGORY=[
    {'name':'背景','label':0,'path':r'{0}/{1}'.format(DEFAULT_IMAGE_ROOT_PATH,0)},
    {'name':'积木','label':1,'path':r'{0}/{1}'.format(DEFAULT_IMAGE_ROOT_PATH,1)},
    {'name':'智能车','label':2,'path':r'{0}/{1}'.format(DEFAULT_IMAGE_ROOT_PATH,2)},
]

