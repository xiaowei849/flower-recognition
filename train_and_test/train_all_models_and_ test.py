import os

file_list = os.listdir('.')
for i in file_list:
    if 'train' in i and 'all' not in i:
        print('当前训练' + i)
        os.system(f'D:/Anaconda3/envs/flower/python.exe E:/python/projects/flower/train_and_test/{i}')
os.system(f'D:/Anaconda3/envs/flower/python.exe E:/python/projects/flower/train_and_test/models_test.py')