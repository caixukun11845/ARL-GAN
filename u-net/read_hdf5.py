
import h5py

# 指定 HDF5 文件的路径
file_path = 'D:/python工程/dataset/cmf/train/ct.h5'  # 将 'your_file.h5' 替换为实际文件路径

# 打开 HDF5 文件，使用 'r' 模式表示只读
with h5py.File(file_path, 'r') as h5_file:
    # 查看文件中的所有组
    print("Keys in the file:", list(h5_file.keys()))

    # 访问特定的数据集（假设文件中有 'dataset' 这个数据集）
    dataset = h5_file['ct'][:]  # 'dataset_name' 替换为实际数据集名称
    print(dataset)  # 打印出数据集的内容






