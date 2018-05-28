import h5py
#HDF5的读取：
f = h5py.File('./model/model.h5','r')
#打开h5文件
#  可以查看所有的主键
for key in f.keys():
    print(f[key].name)
    #print(f[key].shape)
    #print(f[key].value)

import numpy as np
#HDF5的写入：
'''
imgData = np.zeros((30,3,128,256))
f = h5py.File('HDF5_FILE.h5','w')   #创建一个h5文件，文件指针是f
f['data'] = imgData                 #将数据写入文件的主键data下面
f['labels'] = range(100)            #将数据写入文件的主键labels下面
f.close()                            #关闭文件
'''

#HDF5的读取：
'''
f = h5py.File('HDF5_FILE.h5','r')   #打开h5文件
print(f.keys())                          #可以查看所有的主键
a = f['data'][:]                    #取出主键为data的所有的键值
f.close()
'''

