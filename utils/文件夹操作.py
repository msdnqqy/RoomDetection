import os
import shutil
import numpy as np
"""
1.获取某文件夹下所有文件
返回{
【path】：path全路径
【filename】：文件名
}
"""
def get_files_of_folder(path=None):
    if path is None:return;
    paths = os.listdir(path)
    paths_result = []
    for sub in paths:
        paths_result.append({'path': os.path.join(path, sub), 'name':sub})
    return np.array(paths_result)


"""
2.根据文件夹路径不存在则创建文件夹
"""
def check_create_folder(path):
    if  os.path.exists(path):return
    else:os.mkdir(path)

"""
3.递归删除文件夹
"""
def check_remove_folder(path):
    if os.path.exists(path):shutil.rmtree(path)