#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time: 2024/8/28 - 19:13
@Project: STGNN - data_loader
@Author: Yacan Man(曼亚灿)
@Email: manyacan@qq.com
@Website: https://www.manyacan.com
"""
import os
import hues
import pickle
import numpy as np
from data_maker import DataMaker

class StandardScaler(object):
    """
    [模型运行专属] 物理量纲的双向翻译官 (Z-score 缩放器)。
    
    核心定位：它不负责计算统计特征，而是专门承接从 `scaler.pkl` 中加载出来的训练集均值和标准差。
    在训练引擎 (Trainer) 运行时，负责在“现实物理世界（有单位的数据）”与“深度学习抽象世界（无量纲张量）”之间做双向转换。
    """

    def __init__(self, mean, std):
        """
        装载从 scaler.pkl 中提取出的“统计灵魂”。
        注意：这里的 mean 和 std 是严格基于前 70% 训练集计算出的历史稳态基准，绝未受到测试集污染。
        
        :param mean: 训练集各传感器通道的历史基准水位线 (均值)
        :param std:  训练集各传感器通道的历史波动幅度 (标准差)
        """
        self.mean = mean  
        self.std = std    

    def transform(self, data):
        """
        【正向翻译：物理世界 -> 抽象张量】
        动作：去量纲化 (De-dimensionalization)。
        物理意义：将诸如水泵转速 (1500 RPM) 和 管道压力 (2.5 MPa) 这种绝对数值相差悬殊的特征，
        强行按比例压缩到均值为 0、方差为 1 的平滑数学空间中。
        目的：防止数值绝对值大的变量在图神经网络计算时产生“梯度霸权”，确保各个传感器节点被公平对待。
        """
        return (data - self.mean) / self.std  

    def inverse_transform(self, data):
        """
        【逆向翻译：抽象张量 -> 物理世界】 (误差评估环节的绝对核心)
        动作：量纲重构 (Re-dimensionalization)。
        物理意义：当 MTGNN 模型预测出未来 12 步的值时，它吐出的是一堆如 0.05, -0.12 这样的抽象小数。
        如果不做还原，您计算出来的 RMSE 误差毫无工程价值。
        必须通过此函数，将抽象的张量乘上波动幅度、加上基准水位线，精准还原成真实的 RPM 或 MPa。
        目的：确保向学术论文或工程仪表盘输出的 MAE / RMSE 指标，是具有真实物理意义的偏差值。
        """
        return (data * self.std) + self.mean

class DataLoader(object):
    """
    [高维张量调度器] 
    专为大规模工业时序 Seq2Seq 数据设计。负责在训练时将庞大的数据池切割成标准大小的 Batch，
    并提供防崩溃的补齐机制（Padding）和轻量级的随机洗牌机制。
    """
    def __init__(self, xs, ys, batch_size: int = 64, add_pad=True):
        self.batch_size = batch_size  
        self.current_ind = None  
        # =====================================================================
        # 核心设计 1：尾部安全补齐 (Padding 机制)
        # 物理意义：假设您有 1000 个样本，batch_size 是 64。1000 除以 64 等于 15 余 40。
        # 最后剩下的这 40 个样本凑不够一个完整的 Batch。在图神经网络中，很多底层张量运算
        # 严格要求 Batch 维度大小一致，否则极易触发矩阵维度不匹配的致命报错。
        # =====================================================================
        if add_pad:  
            # 计算需要多少个“假样本”才能把最后那个残缺的 Batch 填满 (例如需要 24 个)
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size  
            # 采用“原地踏步”策略：直接把真实数据中的最后一个样本复制 num_padding 份
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)  
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)  
            # 拼接到原数据的尾部，此时总样本数绝对能被 batch_size 完美整除
            xs = np.concatenate([xs, x_padding], axis=0)  
            ys = np.concatenate([ys, y_padding], axis=0)  
        self.sample_size = len(xs)  # 补齐后的绝对安全总样本量
        
        # =====================================================================
        # 核心设计 2：轻量级虚拟索引 (Virtual Indices)
        # =====================================================================
        # 这里只是生成了一串从 0 到 sample_size-1 的数字编号（比如 [0, 1, 2, ..., 1023]）
        self.indices = np.arange(self.sample_size)  
 
        # 算出总共能切出多少个标准的 Batch
        if add_pad:  
            self.num_batch = int(self.sample_size // self.batch_size)  
        else:  
            self.num_batch = int(np.ceil(self.sample_size / self.batch_size))
 
        self.xs = xs  # 挂载真实的输入张量 (X)
        self.ys = ys  # 挂载真实的预测目标 (Y)

    def shuffle(self):
        """
        [训练集专属] 每个 Epoch 开始前的随机洗牌操作。
        工程考量：为了防止模型死记硬背连续时间段的局部特征，必须打乱顺序。
        极其巧妙的是：这里打乱的仅仅是 self.indices 这串单纯的数字编号，
        而绝对没有去移动 self.xs 那个极其庞大的 4D 物理张量，极大地节省了内存和 CPU 开销！
        """
        np.random.shuffle(self.indices)  

    def get_iterator(self):
        """
        [数据传送带] 获取一个 Python 生成器 (Generator)。
        主程序中的 `for x, y in dataloader.get_iterator():` 就是在调用它。
        """
        self.current_ind = 0  

        def _wrapper():
            # =====================================================================
            # 核心设计 3：懒惰切片与 Yield 弹射
            # =====================================================================
            while self.current_ind < self.num_batch:  
                # 算出当前这个 Batch 应该抓取哪段索引
                start_ind = self.batch_size * self.current_ind  
                end_ind = min(self.sample_size, self.batch_size * (self.current_ind + 1))  

                # 根据打乱后的数字编号，去真实的 xs 和 ys 里提取出对应的 64 个样本
                # 张量最终形态: [batch_size (当前批次包含的样本总数), window_size (历史观测时间步长), node_num (工业传感器节点总数), feature_num (单一节点的物理特征维度)]
                batch_indices = self.indices[start_ind:end_ind]  
                x_i = self.xs[batch_indices, ...]  
                y_i = self.ys[batch_indices, ...]  

                # 使用 yield 而不是 return。
                # yield 就像一个弹射器，每次只把这 64 个样本构成的 4D 张量弹给 GPU，
                # 然后函数就会在这里“暂停”，等待主程序处理完这个 Batch，再回来拿下一个。
                # 这就是不爆显存的绝对核心机制。
                yield x_i, y_i  
                
                self.current_ind += 1  # 游标加 1，准备下一次提取

        return _wrapper()

    def get_x_shape(self):
        """
        [模型结构探测器 - 输入端]
        工程使命：主程序在构建 MTGNN 神经网络之前，需要向 DataLoader 询问真实的物理维度信息，
        以确保网络输入层的神经元数量与数据完美对齐。
        
        :return: 返回一个包含四个元素的元组 Tuple，物理形态如下：
            1. num_batch   (总批次量：当前 Epoch 共有多少个 Batch 需要喂给显存，主程序用它来渲染 tqdm 进度条)
            2. window_size (历史观测时间步长：例如您设置的 36 步)
            3. node_num    (传感器节点总数：例如系统中的水泵、阀门总数)
            4. feature_num (单一节点特征维度：如无附加特征则为 1)
        """
        # 语法拆解：
        # 1. self.xs.shape 原本是: [sample_size (样本总数), window_size, node_num, feature_num]
        # 2. self.xs.shape[1:] 是切片操作，扔掉了第 0 维，只保留后面三维的结构。
        # 3. 前面的 * 号是 Python 的“解包 (Unpacking)”操作符。它就像把包裹拆开，
        #    把里面的三个维度拿出来，和前面的 self.num_batch 重新打包成一个全新的四元组。
        return self.num_batch, *self.xs.shape[1:]  

    def get_y_shape(self):
        """
        [模型结构探测器 - 输出端]
        工程使命：主程序需要知道模型最后应该输出多长的预测结果，用于构建最终的全连接层 (End Linear Layer) 
        并计算重构误差 Loss。
        
        :return: 返回一个包含四个元素的元组 Tuple，物理形态如下：
            1. num_batch    (总批次量)
            2. horizon_size (未来预测时间步长：例如您设置的 12 步)
            3. node_num     (传感器节点总数)
            4. feature_num  (单一节点特征维度)
        """
        # 原理同上，精准提取目标 Y 的物理维度
        return self.num_batch, *self.ys.shape[1:]


def load_local_dataset(path: str, batch_size: int = 64, valid_batch_size: int = 64, test_batch_size: int = 64,
                       window_size: int = 36, horizon_size: int = 12, sliding_step: int = 1) -> dict:
    """
    加载本地数据集并生成批量数据加载器。
    支持两种数据源：
    1. 目录路径：包含预处理好的 train.npz、val.npz、test.npz、scaler.pkl 文件
    2. 文件路径：单个数据文件（CSV/PKL/H5），由 DataMaker 在线生成 Seq2Seq 样本
    
    :param path: 数据路径（目录或文件）
    :param batch_size: 训练集批次大小
    :param valid_batch_size: 验证集批次大小
    :param test_batch_size: 测试集批次大小
    :param window_size: 输入时间窗口长度
    :param horizon_size: 输出预测窗口长度
    :param sliding_step: 在线生成滑动窗口样本时使用的步长
    :return: 数据字典，包含：
        - 'train_loader': 训练数据加载器
        - 'val_loader': 验证数据加载器
        - 'test_loader': 测试数据加载器
        - 'scaler': 标准化器对象
        - 'x_train', 'y_train': 训练输入和输出
        - 'x_val', 'y_val': 验证输入和输出
        - 'x_test', 'y_test': 测试输入和输出
    """
    data = {}  # 初始化数据字典
    categories = ['train', 'val', 'test']  # 定义三个数据集类别
    # =====================================================================
    # 如果训练、验证和测试数据集是保存的文件的情况下，直接加载预处理好的 NPZ 文件；如果没有，就在线生成数据。
    # =====================================================================  
    if os.path.isdir(path):  # 如果路径是目录，加载预处理的 NPZ 文件
        # 逐个加载训练、验证和测试数据
        for item in categories:  # 遍历三个数据集类别
            with np.load(os.path.join(path, item + '.npz')) as cat_data:  # 加载对应的 NPZ 文件
                data['x_' + item] = cat_data['x']  # 提取输入特征
                data['y_' + item] = cat_data['y']  # 提取输出标签
                for key in cat_data.files:
                    if key not in ["x", "y"]:
                        data[item + "_" + key] = cat_data[key]

        meta_path = os.path.join(path, "data_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                data["data_meta"] = pickle.load(f)
            data["feature_cols"] = data["data_meta"].get("feature_cols", [])
    elif os.path.isfile(path):  # 如果路径是文件，使用 DataMaker 在线生成数据
        maker = DataMaker(
            path,
            window_size,
            horizon_size,
            sliding_step=sliding_step,
            save=False,
        )  # 创建数据制造器对象（不保存到本地,这里是关键为什么没有将3个数据集保存）
        # 调用 run() 方法生成 Seq2Seq 样本和标准化器
        data['x_train'], data['y_train'], data['x_val'], data['y_val'], data['x_test'], data['y_test'], data[
            'scaler'] = maker.run()  # 解包返回的数据
        for item, extra in maker.generated_extra.items():
            for key, value in extra.items():
                data[item + "_" + key] = value
        data["data_meta"] = maker.data_meta
        data["feature_cols"] = maker.feature_cols
    else:  # 路径既不是目录也不是文件，抛出错误
        raise ValueError('The input data path is not exist.')  # 输入路径不存在

    # =====================================================================
    # 如果是文件的情况下，归一化器 (Scaler) 的反序列化与内存加载
    # =====================================================================
    #  1. 物理探路：用 os.path.join 拼出完整路径（例如 "data/scaler.pkl"），
    # 并检查这个文件在硬盘上是否真实存在。这是防止程序抛出 FileNotFoundError 而崩溃的安全锁。
    if os.path.exists(os.path.join(path, "scaler.pkl")):  
        # 2. 建立通道：使用 with 语句（上下文管理器）安全地打开文件。
        # 注意这里的 'rb' (Read Binary)：因为 .pkl 不是纯文本，而是被冻结的二进制字节流，必须用二进制模式读取。
        # with 语句的好处是，读取完毕后会自动切断文件连接 (f.close())，防止内存泄漏。
        with open(os.path.join(path, "scaler.pkl"), 'rb') as f:  
            # 在终端打印绿色的日志，告知研究员“翻译官”已经就位
            hues.info('已加载归一化类对象 (Scaler).')  
            # 3. 核心复活术 (Deserialization / 反序列化)：
            # pickle.load(f) 会读取这些毫无规律的二进制字节流，并在内存中瞬间将它们
            # 重新拼装成一个活生生的 Python 对象（包含了训练集的均值、标准差以及归一化函数）。
            # 最后，把这个复活的对象挂载到 data 字典的 'scaler' 键上，供后续的验证和测试环节随时调用。
            data['scaler'] = pickle.load(f) 

    # 创建三个数据加载器：训练集、验证集、测试集
    # 输入/输出的 shape：[batch_size, window_size/horizon_size, node_num, feature_num]
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)  # 创建训练数据加载器
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)  # 创建验证数据加载器
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)  # 创建测试数据加载器

    return data  # 返回包含加载器和数据的完整字典


if __name__ == '__main__':
    # 程序主入口（可根据需要添加测试代码）
    pass
