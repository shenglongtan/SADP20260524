#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time: 2024/8/28 - 17:10
@Project: STGNN - data_maker
@Author: Yacan Man(曼亚灿)
@Email: manyacan@qq.com
@Website: https://www.manyacan.com
"""
import os
import hues
import numpy as np
import pandas as pd
import pickle 
from datetime import datetime
from tools import CustomScaler
from typing import Dict, Optional, Tuple, Union





class DataMaker:
    """
    数据预处理类，用于生成 STGNN 模型的训练/验证/测试输入数据。

    功能：
    - 读取原始时序数据文件（CSV、PKL、H5）
    - 可选加入时间特征（小时、星期）
    - 根据滑动窗口生成 Seq2Seq 输入输出对
    - 划分训练、验证、测试集
    - 可选将结果保存为压缩 NPZ 文件
    """

    def __init__(self, file_path: str, window_size: int = 12, horizon_size: int = 12, 
                 sliding_step: int = 1,  # <--- [新增] 滑动步长参数
                 flow_pred: bool = False, hour_as_feature: bool = False, 
                 week_as_feature: bool = False, save: bool = True,
                 output_root: Optional[str] = None):
        """
    初始化 DataMaker 对象。
        :param file_path: 数据文件路径，支持 .csv / .pkl / .h5。
        :param window_size: 过去序列长度，用于模型输入。
        :param horizon_size: 未来预测步数，用于模型输出。
        :param sliding_step: 滑动步长，用于控制滑动窗口的移动距离。
        :param flow_pred: 是否启用专用流量预测偏移逻辑。
        :param hour_as_feature: 是否添加小时信息作为额外特征。
        :param week_as_feature: 是否添加星期信息作为额外特征。
        :param save: 是否将生成的数据保存为 npz 文件。
        :param output_root: 预处理数据输出根目录，默认保存到 ./Save/Datasets。
        """
        self.file_path = file_path
        self.df = None
        self.flow_pred = flow_pred
        self.window_size = window_size
        self.horizon_size = horizon_size
        self.sliding_step = sliding_step
        self.hour_as_feature = hour_as_feature
        self.week_as_feature = week_as_feature
        self.save = save
        self.output_root = output_root or os.path.join(".", "Save", "Datasets")
        self.output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_tag = self._safe_name(os.path.splitext(os.path.basename(file_path))[0])
        self.feature_cols = []
        self.meta_cols = []
        self.attack_col = None
        self.train_col = None
        self.data_meta = {}
        self.generated_extra = {}
        self.output_dir = None

    @staticmethod
    def _find_col(columns, target_name: str):
        """
        在列名中大小写不敏感地查找指定元数据列。
        """
        for col in columns:
            if str(col).strip().lower() == target_name.lower():
                return col
        return None

    @staticmethod
    def _safe_name(value: str) -> str:
        """
        将任意字符串转换为适合目录名的短标签。
        """
        safe = "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in str(value))
        safe = "_".join(part for part in safe.split("_") if part)
        return safe or "dataset"

    def _build_dataset_output_dir(self, split_protocol: str) -> str:
        protocol_tag = "traincol" if split_protocol == "train_column" else "chron"
        dir_name = (
            f"{self.dataset_tag}_w{self.window_size}_h{self.horizon_size}"
            f"_s{self.sliding_step}_{protocol_tag}_{self.output_timestamp}"
        )
        return os.path.join(self.output_root, dir_name)

    @staticmethod
    def _to_binary_series(series: pd.Series, col_name: str) -> pd.Series:
        """
        将 Attack/Train 这类元数据列统一转换为 0/1 序列。
        """
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8)

        text = series.astype(str).str.strip().str.lower()
        if col_name.lower() == "attack":
            positive_values = {"1", "true", "yes", "attack", "attacked", "anomaly", "abnormal"}
        else:
            positive_values = {"1", "true", "yes", "train", "training"}
        return text.isin(positive_values).astype(np.int8)

    def _separate_features_and_metadata(
            self,
            df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        将物理传感器特征与实验元数据列分离，避免 Attack/Train 泄漏到模型输入。
        """
        self.attack_col = self._find_col(df.columns, "Attack")
        self.train_col = self._find_col(df.columns, "Train")
        self.meta_cols = [col for col in [self.attack_col, self.train_col] if col is not None]

        feature_cols = [col for col in df.columns if col not in self.meta_cols]
        non_numeric_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            hues.warn(f"以下非数值列不会进入模型特征: {non_numeric_cols}")
            feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
        if not feature_cols:
            raise ValueError("未找到可用于模型训练的数值型物理特征列。")

        self.feature_cols = feature_cols
        feature_df = df[feature_cols].astype(np.float64)
        attack_series = self._to_binary_series(df[self.attack_col], "Attack") if self.attack_col is not None else None
        train_series = self._to_binary_series(df[self.train_col], "Train") if self.train_col is not None else None

        hues.info(f"模型物理特征列数量: {len(self.feature_cols)}，已排除元数据列: {self.meta_cols}")
        return feature_df, attack_series, train_series

    def _build_offsets(self) -> Tuple[np.array, np.array]:
        """
        构建 Seq2Seq 输入/输出的时间偏移表。
        """
        if self.flow_pred:
            x_offset = np.sort(np.array([-7 * 24, -3 * 24, -2 * 24, -1 * 24, -3, -2, -1]))
        else:
            x_offset = np.sort(
                np.concatenate((np.arange(-(self.window_size - 1), 1, 1),))
            )
        y_offset = np.sort(np.arange(1, (self.horizon_size + 1), 1))
        return x_offset, y_offset

    def _split_dataframe(
            self,
            feature_df: pd.DataFrame,
            attack_series: Optional[pd.Series],
            train_series: Optional[pd.Series],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Optional[pd.Series]], str]:
        """
        生成训练/验证/测试原始行划分。

        若存在 Train 列，优先采用工业异常检测常见协议：
        Train==1 作为训练/验证候选段，Train==0 作为测试段。
        """
        if train_series is not None:
            train_val_mask = train_series == 1
            test_mask = train_series == 0
            if train_val_mask.any() and test_mask.any():
                train_val_df = feature_df.loc[train_val_mask].copy()
                test_df = feature_df.loc[test_mask].copy()
                train_val_attack = attack_series.loc[train_val_mask].copy() if attack_series is not None else None
                test_attack = attack_series.loc[test_mask].copy() if attack_series is not None else None

                split_idx = int(len(train_val_df) * 0.7)
                if split_idx <= 0 or split_idx >= len(train_val_df):
                    raise ValueError("Train==1 数据段过短，无法进一步划分训练集和验证集。")

                split_dfs = {
                    "train": train_val_df.iloc[:split_idx].copy(),
                    "val": train_val_df.iloc[split_idx:].copy(),
                    "test": test_df,
                }
                split_attacks = {
                    "train": train_val_attack.iloc[:split_idx].copy() if train_val_attack is not None else None,
                    "val": train_val_attack.iloc[split_idx:].copy() if train_val_attack is not None else None,
                    "test": test_attack,
                }
                hues.info("检测到 Train 列：采用 Train==1 训练/验证、Train==0 测试的异常检测协议。")
                return split_dfs, split_attacks, "train_column"

            hues.warn("Train 列存在，但无法同时形成 Train==1 和 Train==0 两个数据段，将回退为时间顺序划分。")

        num_rows = len(feature_df)
        num_train = round(num_rows * 0.7)
        num_val = round(num_rows * 0.1)
        if num_train <= 0 or num_val <= 0 or num_train + num_val >= num_rows:
            raise ValueError("数据长度过短，无法按 70%/10%/20% 生成训练、验证和测试集。")

        split_dfs = {
            "train": feature_df.iloc[:num_train].copy(),
            "val": feature_df.iloc[num_train:num_train + num_val].copy(),
            "test": feature_df.iloc[num_train + num_val:].copy(),
        }
        split_attacks = {
            "train": attack_series.iloc[:num_train].copy() if attack_series is not None else None,
            "val": attack_series.iloc[num_train:num_train + num_val].copy() if attack_series is not None else None,
            "test": attack_series.iloc[num_train + num_val:].copy() if attack_series is not None else None,
        }
        hues.info("未使用 Train 列协议：采用严格时间顺序 70%/10%/20% 划分。")
        return split_dfs, split_attacks, "chronological_70_10_20"

    def run(self) -> Union[None, Tuple[np.array, np.array, np.array, np.array, np.array, np.array, CustomScaler]]:
        """
        DataMaker 的主入口函数，执行完整的数据预处理流程。

        返回值：当 save=True 时返回 None；当 save=False 时返回数据和 scalers 对象。
        - x_train: 训练集输入，形状 [num_train_samples, window_size, node_num, feature_num]
        - y_train: 训练集输出，形状 [num_train_samples, horizon_size, node_num, feature_num]
        - x_val: 验证集输入，形状 [num_val_samples, window_size, node_num, feature_num]
        - y_val: 验证集输出，形状 [num_val_samples, horizon_size, node_num, feature_num]
        - x_test: 测试集输入，形状 [num_test_samples, window_size, node_num, feature_num]
        - y_test: 测试集输出，形状 [num_test_samples, horizon_size, node_num, feature_num]
        - scalers: 归一化/标准化器对象（仅在 save=False 时返回）
        """
        hues.info("  Generating the Input data of Model  ".center(70, '='))

        raw_df = self.get_df(self.file_path)
        self.df, attack_series, train_series = self._separate_features_and_metadata(raw_df)
        split_dfs, split_attacks, split_protocol = self._split_dataframe(
            self.df,
            attack_series,
            train_series,
        )
        if self.save:
            self.output_dir = self._build_dataset_output_dir(split_protocol)

        # 只允许训练段参与标准化统计量估计，避免验证/测试分布污染训练基准。
        scalers = CustomScaler(split_dfs["train"].values)
        x_offset, y_offset = self._build_offsets()

        x_train, y_train, extra_train = self.create_input_data(
            split_dfs["train"], scalers, x_offset, y_offset, split_attacks["train"]
        )
        x_val, y_val, extra_val = self.create_input_data(
            split_dfs["val"], scalers, x_offset, y_offset, split_attacks["val"]
        )
        x_test, y_test, extra_test = self.create_input_data(
            split_dfs["test"], scalers, x_offset, y_offset, split_attacks["test"]
        )

        self.generated_extra = {
            "train": extra_train,
            "val": extra_val,
            "test": extra_test,
        }
        self.data_meta = {
            "feature_cols": self.feature_cols,
            "meta_cols": self.meta_cols,
            "attack_col": self.attack_col,
            "train_col": self.train_col,
            "split_protocol": split_protocol,
            "window_size": self.window_size,
            "horizon_size": self.horizon_size,
            "sliding_step": self.sliding_step,
            "raw_rows": len(raw_df),
            "split_rows": {name: len(df) for name, df in split_dfs.items()},
            "output_dir": self.output_dir,
        }

        hues.info(f"训练集 x: {x_train.shape}, y: {y_train.shape}")
        hues.info(f"验证集 x: {x_val.shape}, y: {y_val.shape}")
        hues.info(f"测试集 x: {x_test.shape}, y: {y_test.shape}")
        hues.info("  Done!!!  ".center(70, '='))
        # =====================================================================
        # 5. 最终发货工序：根据 save 参数决定是“落盘归档”还是“内存直传”
        # =====================================================================
        if self.save:
            # 【分支 A：硬盘归档模式】
            # 适用于离线生成标准数据集，供跨设备共享或后续重复读取
            # 检查并自动创建用于存放数据的本地物理文件夹
            self.check_folder(self.output_dir)  
            # 遍历刚才切断剥离出来的三个数据集合
            for cat in ["train", "val", "test"]:
                # 核心动态变量抓取：locals() 会以字典形式读取当前空间内的所有局部变量。
                # 例如：当 cat 循环到 "train" 时，locals()["x_train"] 等价于直接调取前面的 x_train 变量。
                # 这种写法极其精简，避免了冗长死板的 if-elif 连环判断。
                _x, _y = locals()["x_" + cat], locals()["y_" + cat]  
                _extra = self.generated_extra.get(cat, {})
                # 在终端实时打印当前正在保存的数据集类型及其张量维度，供二次核对
                hues.info(cat, "x:", _x.shape, ", y:", _y.shape, '.')
                # 高压落盘机制：由于 36 步长的高维特征矩阵体积极其庞大，用普通的 CSV 会轻易撑爆硬盘。
                # 这里强制采用 np.savez_compressed 函数，将 x 和 y 像打包 Zip 一样，
                # 封装成极高压缩率的 .npz 专用文件并写入硬盘。
                save_kwargs = {"x": _x, "y": _y}
                save_kwargs.update(_extra)
                np.savez_compressed(
                    os.path.join(self.output_dir, "%s.npz" % cat),
                    **save_kwargs
                )  
            # 2. 【关键修复】单独打包保存 Scaler 对象 (带灵魂的归一化器)
            scaler_path = os.path.join(self.output_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:  # wb: 二进制写入模式
                pickle.dump(scalers, f)
            hues.info(f"Scaler object saved to: [{scaler_path}]")

            meta_path = os.path.join(self.output_dir, "data_meta.pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(self.data_meta, f)
            hues.info(f"Data metadata saved to: [{meta_path}]")
            
            # 归档完毕，打印绿色的成功日志并附上文件绝对路径
            hues.success(f'The file was successfully written in: [{self.output_dir}].')
            # 数据已安全落盘，不再需要占用内存传输，因此直接向主程序返回空值
            return None
        else:
            # 【分支 B：内存直传模式】(实验训练实际走的最高效分支)
            # 随用随切，切完直传，算完即焚。彻底省去硬盘 I/O 读写开销，是保障整体训练速度的核心。
            # 像接力棒一样，将 6 个切好的高维张量以及“防穿越”计算出来的 scalers 归一化对象，
            # 原封不动地直接返回给上一层，供引擎即刻投入运算。
            return x_train, y_train, x_val, y_val, x_test, y_test, scalers

    def create_input_data(
            self,
            feature_df: pd.DataFrame,
            scalers: CustomScaler,
            x_offset: np.array,
            y_offset: np.array,
            attack_series: Optional[pd.Series] = None,
    ) -> Tuple[np.array, np.array, Dict[str, np.array]]:
        """
        核心功能：像“切土豆片”一样，将整条长长的时间序列，按规定的窗口和步长，切成一个个供模型学习的样本块 (Seq2Seq)。
        """
        # 1. 获取基础维度：获取时间轴总长度 (sample_num) 和 传感器节点总数 (node_num)
        sample_num, node_num = feature_df.shape  
        if sample_num == 0:
            raise ValueError("当前数据划分为空，无法生成滑动窗口样本。")

        min_t = abs(min(x_offset))
        max_t = sample_num - abs(max(y_offset))
        epoch_len = len(range(min_t, max_t, self.sliding_step))
        if epoch_len <= 0:
            raise ValueError(
                f"当前数据划分长度为 {sample_num}，不足以生成 window_size={self.window_size}, "
                f"horizon_size={self.horizon_size} 的样本。"
            )

        # 只有物理传感器矩阵会进入标准化，Attack/Train/Timestamp 不参与模型输入。
        scaled_values = scalers.transform(feature_df.values).astype(np.float32)
        # 2. 升维装载原始数据：把原本的 2D 矩阵 (时间, 传感器) 变成 3D 矩阵 (时间, 传感器, 特征)
        # 这里的初始特征数是 1（即传感器原始的物理读数，比如水压、流量等）
        data = np.expand_dims(scaled_values, axis=-1)
        data_list = [data]  
        # 3. 动态特征拼接 (可选)：如果启用了时间特征，就把它们作为“新的特征通道”像夹心饼干一样拼到最后面
        if self.hour_as_feature:
            if not isinstance(feature_df.index, pd.DatetimeIndex):
                raise ValueError("启用 hour_as_feature=True 时，数据索引必须是 DatetimeIndex。")
            time_ind = (
                feature_df.index.values - feature_df.index.values.astype("datetime64[D]")
            ) / np.timedelta64(1, "D")
            time_in_day = np.repeat(time_ind.astype(np.float32)[:, None, None], node_num, axis=1)
            data_list.append(time_in_day)  # 拼入“一天中的哪一小时”
        if self.week_as_feature:
            if not isinstance(feature_df.index, pd.DatetimeIndex):
                raise ValueError("启用 week_as_feature=True 时，数据索引必须是 DatetimeIndex。")
            day_in_week = np.zeros(shape=(sample_num, node_num, 7))
            day_in_week[np.arange(sample_num), :, feature_df.index.dayofweek] = 1
            data_list.append(day_in_week)  # 拼入“一周中的哪一天”
        # 将基础读数和时间特征在最后一个维度 (axis=-1) 捏合在一起。
        # 最终 data 变成了完整的 3D 大矩阵：[总时间步, 节点数, 特征数]
        data = np.concatenate(data_list, axis=-1)
        # =====================================================================
        # 4. 开始滑动窗口切片 (核心动作)
        # =====================================================================
        # 滑动窗口切片：在安全边界内，以 sliding_step 为步长，依次切出输入 X 和输出 Y 的样本块
        x, y = [], []
        extras = {}
        y_attack_window, y_attack_point, y_time = [], [], []
        attack_values = attack_series.to_numpy(dtype=np.int8) if attack_series is not None else None
        if isinstance(feature_df.index, pd.DatetimeIndex):
            timestamp_values = feature_df.index.values.astype("datetime64[ns]")
        else:
            timestamp_values = feature_df.index.astype(str).to_numpy()
        # 滑动步长 (sliding_step)：决定相框每次往前跳跃几个时间点。设为 12 意味着样本间重叠度降低，大幅减少数据量
        # 启动滑动循环：游标 t 在安全范围内，每次跳跃 sliding_step 步
        for t in range(min_t, max_t, self.sliding_step):
            x_t = data[t + x_offset, ...]  # 截取过去的连续时刻，作为输入特征 X
            y_t = data[t + y_offset, ...]  # 截取未来的连续时刻，作为预测目标 Y
            x.append(x_t)
            y.append(y_t)
            if attack_values is not None:
                attack_window = attack_values[t + y_offset]
                y_attack_point.append(attack_window)
                y_attack_window.append(np.max(attack_window))
            y_time.append(timestamp_values[t + y_offset])
            
        # 5. 数据打包：把刚才切出来的一大堆散碎片段，像叠报纸一样堆叠成统一的 4D 张量送给 PyTorch
        x = np.stack(x, axis=0)  # 最终形态：[窗口样本数, 历史窗口长, 传感器数, 特征数]
        y = np.stack(y, axis=0)  # 最终形态：[窗口样本数, 预测窗口长, 传感器数, 特征数]

        if attack_values is not None:
            extras["y_attack_window"] = np.asarray(y_attack_window, dtype=np.int8)
            extras["y_attack_point"] = np.stack(y_attack_point, axis=0).astype(np.int8)
        extras["y_time"] = np.stack(y_time, axis=0)

        hues.info(f'The length of the generated Graph Seq2Seq Io Data is: {epoch_len}.')
        
        return x, y, extras
    
    @staticmethod
    def get_df(file_path: str) -> pd.DataFrame:
        """
        万能数据读取适配器。
        根据传入的文件后缀名，自动选择合适的引擎将数据加载到内存中，并转化为标准的 DataFrame 格式。
        """
        # 1. 拆解路径：将诸如 "data/metr-la.h5" 拆分为路径和后缀 ".h5"
        _, file_extension = os.path.splitext(file_path)  

        # 2. 格式路由分发
        if file_extension == '.csv':
            # 读取最常见的纯文本表格格式
            df = pd.read_csv(file_path)  
            
        elif file_extension in ['.pkl', '.pickle']:
            # 读取 Python 原生的序列化二进制文件（读写速度比 CSV 快得多）
            df = pd.read_pickle(file_path)  
               
        elif file_extension in ['.h5', '.hdf5']:
            # 读取 HDF5 格式（专为海量科学计算数据设计的高性能工业级格式）
            df = pd.read_hdf(file_path)  
            
        else:
            # 安全防线：如果不认识这个后缀，直接报错拦截，防止程序往下瞎跑
            raise ValueError(f"Can't support this type: {file_extension}.")  

        # 【关键时序矫正】如果数据列中包含 'Timestamp'，强制将其从“普通数据”提升为“时间索引”。
        # 这是时间序列模型对齐特征、防止时间步错乱的根本前提。
        timestamp_col = DataMaker._find_col(df.columns, "Timestamp")
        if timestamp_col is not None:
            df = df.set_index(timestamp_col)
              
        return df

    @staticmethod
    def check_folder(save_file: str):
        """
        目录安全检查与创建工具。
        在向硬盘写入数据前，确保目标“仓库”物理存在，防止引发 FileNotFoundError 导致程序中途崩溃。
        """
        # 如果操作系统在硬盘上找不到这个路径
        if not os.path.exists(save_file):  
            # 递归建立完整的目录树（如果父目录不存在也会一并创建）
            os.makedirs(save_file)  
            # 在控制台打印绿色的成功建档提示
            hues.info(f"Successfully Created the folder: ['{save_file}'].")

if __name__ == '__main__':
    # 实例化 DataMaker 对象，配置数据路径和参数，并执行数据生成流程，生成的结果将直接保存到指定目录下的 .npz 文件中，供后续训练使用。
    #如果不想保存到本地，可以将 save=False，直接返回数据和 scalers 对象供内存使用。
    #如果外部调用这个模块时，想要直接获取数据而不是保存文件，也可以将 save=False，这样 run() 方法会返回切好的数据和 scalers 对象，供调用者直接使用。
    maker = DataMaker(
        "D:/大学/博士论文/GNN部分/07_RA-STGNN/Data/SWat/swat_data_10s.pkl",  # 数据文件路径
        window_size=36,  # 输入窗口长度
        horizon_size=12,  # 预测窗口长度
        flow_pred=False,  # 不使用流量预测模式
        hour_as_feature=False,  # 不加入小时特征
        week_as_feature=False,  # 不加入星期特征
        save=True  # 保存为文件
    )
    maker.run()  # 执行数据生成流程
