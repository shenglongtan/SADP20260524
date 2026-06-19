#!/usr/bin/env python3  # [语法] 指定类 Unix 系统使用 Python 3 解释器执行此脚本
# -*- coding: utf-8 -*-  # [语法] 声明文件编码为 UTF-8，确保能正确解析中文注释

# ==========================================
# 模块一：环境依赖与全局配置
# 功能：导入必要的数据处理、评估指标与可视化库，并初始化绘图参数
# ==========================================
import json  # [语法] 导入内置 json 库。[功能] 用于读取和解析模型运行的超参数配置文件
import pickle  # [语法] 导入内置 pickle 库。[功能] 用于反序列化加载以 .pkl 格式存储的真实标签数据
import warnings  # [语法] 导入 warnings 库。[功能] 用于控制终端的警告信息输出
from pathlib import Path  # [语法] 从 pathlib 导入 Path 类。[功能] 提供面向对象的文件路径操作，跨平台兼容性更好
from typing import Tuple, Dict, Optional, List  # [语法] 导入类型提示。[功能] 标注函数参数和返回值的具体数据类型，提升代码可读性

import numpy as np  # [语法] 导入 NumPy 并简写为 np。[功能] 提供高性能的多维数组（矩阵）计算支持
import pandas as pd  # [语法] 导入 Pandas 并简写为 pd。[功能] 用于构建和分析二维表格数据（DataFrame）
import matplotlib.pyplot as plt  # [语法] 导入 Matplotlib 的绘图模块。[功能] 用于后续生成异常对比的可视化图表
from sklearn.metrics import (  # [语法] 从 scikit-learn 机器学习库中导入多个评估函数
    confusion_matrix, precision_score, recall_score, f1_score,  # [功能] 用于计算混淆矩阵、精确率、召回率、综合F1分数
    roc_auc_score, average_precision_score  # [功能] 用于计算 ROC 曲线下面积和 PR 曲线下面积（评价数据不平衡时的性能）
)

warnings.filterwarnings('ignore')  # [语法] 调用过滤器函数。[功能] 全局屏蔽运行时的非致命警告（如版本废弃提示），保持终端输出整洁

plt.rcParams.update({  # [语法] 更新 matplotlib 的全局配置字典。[功能] 统一设定生成的图表样式
    'font.size': 12,  # [语法] 键值对赋值。[功能] 设置图表全局基础字体大小为 12
    'font.family': ['Times New Roman', 'SimSun'],  # [语法] 列表赋值。[功能] 优先使用新罗马英文字体，中文回退使用宋体
    'figure.dpi': 120  # [语法] 键值对赋值。[功能] 设置图像分辨率为 120 像素/英寸，保证图表清晰度
})


# ==========================================
# 模块二：核心管道类定义与数据初始化
# 功能：封装异常检测的后处理流程，负责加载预测张量、模型参数及基准标签
# ==========================================
class AnomalyDetectionPipeline:  # [语法] 定义一个类。[功能] 将数据加载、计算得分、点调整等所有后处理步骤封装为一个管道对象
    def __init__(self, save_folder: Path, label_path: Optional[Path] = None):  # [语法] 类的构造函数（初始化方法）。[功能] 接收数据保存目录和可选的标签路径
        """初始化管道"""
        self.save_folder = Path(save_folder)  # [语法] 实例化 Path 对象并存入实例属性。[功能] 锁定模型输出文件所在的根目录
        self.label_path = Path(label_path) if label_path else None  # [语法] 三元条件表达式。[功能] 若传入了标签路径则转换为 Path 对象，否则保持为空
        
        # [功能] 调用内部私有方法，按顺序加载执行异常分析所需的三大基础数据：预测结果、参数、真实标签
        self._load_predictions()
        self._load_parameters()
        self._load_labels()

    def _load_predictions(self):
        """加载已保存的预测结果"""
        y_true_path = self.save_folder / 'y_true.npy'  # [语法] 使用 / 运算符拼接路径。[功能] 获取真实观测值的本地路径
        y_pred_path = self.save_folder / 'y_pred.npy'  # [语法] 同上。[功能] 获取模型预测值的本地路径
        
        if not y_true_path.exists() or not y_pred_path.exists():  # [语法] 逻辑或运算判断文件是否存在。[功能] 拦截缺失文件的错误
            raise FileNotFoundError(f"Missing prediction files in {self.save_folder}")  # [语法] 抛出异常。[功能] 终止程序并提示错误位置
        
        self.y_true = np.load(y_true_path)  # [语法] np.load 加载二进制数据。[功能] 将真实的 3D 观测张量读入内存
        self.y_pred = np.load(y_pred_path)  # [语法] 同上。[功能] 将模型的 3D 预测张量读入内存
        
        assert self.y_true.shape == self.y_pred.shape, "Shape mismatch"  # [语法] assert 断言语句。[功能] 强制校验真实值和预测值的维度矩阵完全一致
        assert self.y_true.ndim == 3, f"Expected 3D array"  # [语法] 检查数组维度属性 ndim。[功能] 确保数据是标准的时空三维格式
        
        # [语法] 序列解包赋值。[功能] 提取 3D 张量的三个维度大小：样本总数(N)、传感器/变量数(V)、时间步长(H)
        self.num_samples, self.num_vars, self.horizon_size = self.y_true.shape
        print(f"✓ Loaded predictions: {self.num_samples} samples × {self.num_vars} vars × {self.horizon_size} horizon")

    def _load_parameters(self):
        """加载模型参数配置"""
        params_path = self.save_folder / 'Parameters.json'
        
        with params_path.open('r', encoding='utf-8') as f:  # [语法] with 上下文管理器以只读/UTF8打开文件。[功能] 安全读取配置文件，自动释放文件句柄
            self.params = json.load(f)  # [语法] json.load 将文件流反序列化为字典。[功能] 获取模型的超参数字典
        
        self.window_size = self.params.get('window_size', 96)  # [语法] dict.get 安全获取键值。[功能] 提取时序滑动窗口大小，缺省默认 96
        print(f"✓ Loaded parameters: window_size={self.window_size}, horizon_size={self.horizon_size}")

    def _load_labels(self):
        """加载标签数据（如果可用）"""
        if self.label_path and self.label_path.exists():  # [语法] 短路逻辑检查对象是否非空且路径存在
            try:  # [语法] try-except 异常捕获块。[功能] 防止因标签文件损坏导致整个流水线崩溃
                with open(self.label_path, 'rb') as f:  # [语法] rb 模式。[功能] 以二进制只读方式打开 pkl 文件
                    data = pickle.load(f)  # [语法] pickle.load 反序列化。[功能] 将二进制字节流还原为 Pandas 或 Numpy 原始对象
                    self.labels_df = data  # [功能] 将标签数据挂载到类的实例属性上供全局使用
                    print(f"✓ Loaded labels from {self.label_path}")
            except Exception as e:
                print(f"⚠ Failed to load labels: {e}")
                self.labels_df = None  # [功能] 发生异常时进行降级处理，将标签置空
        else:
            self.labels_df = None
            print("⚠ No labels provided (will use unsupervised threshold selection)")


# ==========================================
# 模块三：异常分值计算
# 功能：将 [N, V, H] 的重构误差通过双重降维（时间步与变量维度）及归一化，转化为一维的一致性异常分值 [N]
# ==========================================
    def compute_anomaly_scores(self, horizon_reduce: str = 'max', var_reduce: str = 'max') -> np.ndarray:
        # [语法] np.abs 求绝对值矩阵。[功能] 计算真实数据与预测数据之间的重构残差绝对值，形状保持 [N, V, H]
        error = np.abs(self.y_true - self.y_pred)
        
        # --- 第一步：时间步维度降维 (Horizon Reduction) ---
        if horizon_reduce == 'max':
            var_error = error.max(axis=2)  # [语法] axis=2 沿第三维求最大值。[功能] 提取预测窗口内各个变量偏离最严重的时间点，形状缩减为 [N, V]
        elif horizon_reduce == 'mean':
            var_error = error.mean(axis=2)  # [语法] axis=2 沿第三维求均值。[功能] 提取预测窗口内各个变量的平均偏离程度，形状缩减为 [N, V]
        else:
            raise ValueError(f"Unknown horizon_reduce: {horizon_reduce}")
        
        # --- 第二步：多变量特征归一化 (Z-score Normalization) ---
        mu = var_error.mean(axis=0, keepdims=True)  # [语法] axis=0 沿样本维求均值，keepdims=True 保持二维形状。[功能] 计算每个传感器的历史误差均值 [1, V]
        sigma = var_error.std(axis=0, keepdims=True)  # [语法] axis=0 沿样本维求标准差。[功能] 计算每个传感器的误差波动率 [1, V]
        sigma = np.where(sigma == 0, 1e-8, sigma)  # [语法] np.where 数组条件替换。[功能] 拦截标准差为0的平稳常数特征，防止后续触发除以0的报错
        norm_var_error = (var_error - mu) / sigma  # [语法] 数组广播计算。[功能] 消除不同传感器之间量纲和数值范围的差异，将误差映射到同一标准尺度 [N, V]
        
        # --- 第三步：变量维度降维 (Variable Reduction) ---
        if var_reduce == 'max':
            scores = norm_var_error.max(axis=1)  # [语法] axis=1 沿变量维求最大值。[功能] 木桶效应，以当前时刻最异常的那个传感器的归一化偏差，作为系统整体异常得分 [N]
        elif var_reduce == 'mean':
            scores = norm_var_error.mean(axis=1)  # [语法] axis=1 沿变量维求均值。[功能] 评估多传感器的系统整体平均偏离程度 [N]
        else:
            raise ValueError(f"Unknown var_reduce: {var_reduce}")
        
        self.scores = scores  # [功能] 保存最终生成的一维异常得分 [N]
        self.norm_var_error = norm_var_error  # [功能] 保存归一化矩阵，为后续“变量贡献度分析”提供溯源依据
        self.var_error = var_error
        print(f"✓ Computed anomaly scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        return scores


# ==========================================
# 模块四：阈值寻优策略
# 功能：通过有监督网格搜索或无监督统计方法，在连续的异常得分上划定一条判定界限
# ==========================================
    def select_threshold(self, method: str = 'auto', percentile: float = 99.0, grid_size: int = 200) -> Tuple[float, Optional[Dict]]:
        if not hasattr(self, 'scores'):  # [语法] hasattr 检查对象是否包含特定属性。[功能] 拦截逻辑顺序错误，确保已先计算得分
            raise ValueError("Must call compute_anomaly_scores() first")
        
        # 根据有无标签动态分流路由
        if method == 'auto' and self.labels_df is not None:
            print("\n═ 阈值F1优化 (Grid Search) ═")
            threshold, metrics = self._optimize_threshold_by_f1(grid_size=grid_size)  # [功能] 进入有监督分支：执行最大化 F1 指标的网格搜索
        elif method == 'percentile' or (method == 'auto' and self.labels_df is None):
            print(f"\n═ 阈值百分位数选择 (Percentile={percentile}) ═")
            threshold, metrics = self._select_threshold_percentile(percentile)  # [功能] 进入无监督分支：基于数据分布的百分位进行硬截断
        elif method == 'mean_std':
            print("\n═ 阈值均值±标差选择 ═")
            threshold = float(self.scores.mean() + 3 * self.scores.std())  # [语法] 浮点型转换。[功能] 经典的无监督 3-Sigma 原则划分
            metrics = {'method': 'mean_std', 'threshold': threshold}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.threshold = threshold
        self.threshold_method = method
        return threshold, metrics

    def _optimize_threshold_by_f1(self, grid_size: int = 200) -> Tuple[float, Dict]:
        """通过网格搜索穷举寻找使全局 F1 分数最高的最优阈值"""
        if self.labels_df is None:
            raise ValueError("Labels required for F1 optimization")
        
        true_labels = self._extract_labels_for_samples()  # [功能] 提取并对齐基准真实标签（将字符串列名或冗余长度处理为干净的二值标签序列）
        if true_labels is None or len(true_labels) != len(self.scores):
            raise ValueError("Label length mismatch")
        
        score_min, score_max = self.scores.min(), self.scores.max()
        tau_grid = np.linspace(score_min, score_max, grid_size)  # [语法] np.linspace 生成等差数列。[功能] 在得分的最小值和最大值之间打上指定分辨率的离散网格点
        
        best_f1, best_tau, best_metrics = -1, None, {}
        
        for tau in tau_grid:  # [语法] for 循环遍历每一个候选界线
            pred_labels = (self.scores > tau).astype(int)  # [语法] 比较运算产生布尔数组，astype(int) 强制转为 0/1 整型。[功能] 生成当前候选界线下的二分类预测序列
            
            if pred_labels.sum() == 0 or pred_labels.sum() == len(pred_labels):  # [语法] sum 求和统计 1 的个数。[功能] 过滤掉所有样本全被划为正常或全划为异常的无效界线边界
                continue
            
            f1 = f1_score(true_labels, pred_labels, zero_division=0)  # [语法] 调用 sklearn 的 f1_score。[功能] 计算真值与预测值之间的 F1 精度（解决类别不平衡问题的综合评价）
            
            if f1 > best_f1:  # [语法] 数值比较更新。[功能] 维护历史最优记录
                best_f1 = f1
                best_tau = tau
                best_metrics = {
                    'threshold': tau, 'f1': f1,
                    'precision': precision_score(true_labels, pred_labels, zero_division=0),
                    'recall': recall_score(true_labels, pred_labels, zero_division=0),
                }
        
        print(f"  Best threshold: {best_tau:.6f}")
        print(f"  Best F1: {best_metrics['f1']:.4f}")
        return best_tau, {'method': 'f1_optimization', **best_metrics}  # [语法] ** 解包字典操作。[功能] 合并方法名称字段和统计指标字典后返回

    def _select_threshold_percentile(self, percentile: float = 99.0) -> Tuple[float, Dict]:
        threshold = float(np.percentile(self.scores, percentile))  # [语法] np.percentile 计算指定分位数。[功能] 假设前 (100-percentile)% 的极值是异常
        pred_labels = (self.scores > threshold).astype(int)
        metrics = {
            'method': 'percentile', 'percentile': percentile, 'threshold': threshold,
            'anomaly_ratio': float(pred_labels.mean()), 'anomaly_count': int(pred_labels.sum())
        }
        return threshold, metrics

    def _extract_labels_for_samples(self) -> Optional[np.ndarray]:
        """标签自适应提取器"""
        if self.labels_df is None: return None
        
        if isinstance(self.labels_df, pd.DataFrame):  # [语法] isinstance 类型判定。[功能] 兼容处理直接读取为二维表格格式的数据集（如 WADI/SWaT）
            # [语法] 列表推导式与条件判定的组合。[功能] 探测并定位 DataFrame 中究竟是哪一列代表异常状态 (支持 'Attack', 'Label' 或模糊匹配)
            if 'Attack' in self.labels_df.columns:
                col = 'Attack'
            elif 'Label' in self.labels_df.columns:
                col = 'Label'
            else:
                label_cols = [c for c in self.labels_df.columns if 'label' in c.lower() or 'attack' in c.lower()]
                if not label_cols: return None
                col = label_cols[0]
            
            labels = self.labels_df[col].values[:self.num_samples]  # [语法] .values 提取 numpy 数组，切片 [:N] 截取。[功能] 剥离表结构，强制与模型预测输出的时间跨度进行对齐截断
            return (labels != 0).astype(int)  # [语法] 逻辑判断转整型。[功能] 将多类别错误类型（如有）统一二值化归一为标准异常（1）与正常（0）
            
        elif isinstance(self.labels_df, np.ndarray):  # [功能] 兼容处理直接传入的一维标签数组格式
            return (self.labels_df[:self.num_samples] != 0).astype(int)
        
        return None


# ==========================================
# 模块五：时域点调整（Point Adjustment, PA）
# 功能：工业时序异常检测的标准容错策略。由于真实攻击/故障的传播存在延迟，
#       只要在指定的邻域窗口内命中了任意异常，就判定整个窗口区间报警成功。
# ==========================================
    def apply_point_adjustment(self, pred_labels: np.ndarray, delay: Optional[int] = None) -> np.ndarray:
        if delay is None:
            delay = self.horizon_size - 1  # [功能] 如果未自定义延迟步数，默认允许的扩充容错范围为模型所能预测的时间步长减 1
        
        adjusted = pred_labels.copy()  # [语法] np.copy 深拷贝。[功能] 保护原始预测标签序列不被污染，在副本上进行覆盖扩充操作
        
        for i in range(len(pred_labels)):  # [语法] for 循环配合 range 枚举索引。[功能] 遍历时间轴上的每一个样本点
            start = max(0, i - delay)  # [语法] max 限制下界。[功能] 计算当前点的左侧关联起点，防止索引变为负数越界
            end = min(len(pred_labels), i + delay + 1)  # [语法] min 限制上界。[功能] 计算当前点的右侧关联终点（左闭右开区间），防止越出数组上限
            
            if np.any(pred_labels[start:end]):  # [语法] 数组切片与 np.any 逻辑存在性判断。[功能] 扫描该邻域窗口，如果含有任何一个预测出的“1”（异常点）
                adjusted[start:end] = 1  # [语法] 切片批量赋值。[功能] 将这整段时域局部连通区强行标记/扩展为“1”，完成 PA 点调整的修正扩张
        
        self.pred_labels_adjusted = adjusted
        print(f"✓ Point adjustment applied (delay={delay})")
        print(f"  Original anomalies: {pred_labels.sum()}, Adjusted anomalies: {adjusted.sum()}")
        return adjusted


# ==========================================
# 模块六：分类性能度量与根因分析
# 功能：产出 PA 调整前后的指标对比矩阵表，并评估各项传感器的致错贡献
# ==========================================
    def generate_evaluation_report(self, pred_labels_before: np.ndarray, pred_labels_after: np.ndarray) -> pd.DataFrame:
        if self.labels_df is None: return None
        true_labels = self._extract_labels_for_samples()
        if true_labels is None: return None
        
        # [功能] 分别传入调整前与调整后的标签向量进行指标运算
        metrics_before = self._compute_metrics(true_labels, pred_labels_before, "点调整前")
        metrics_after = self._compute_metrics(true_labels, pred_labels_after, "点调整后")
        
        # [语法] 字典嵌套列表构建 DataFrame。[功能] 拼接生成易读的对比数据表
        report_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'True Positives', 'False Positives', 'False Negatives'],
            '点调整前': [
                f"{metrics_before['precision']:.4f}",  # [语法] f-string 配合 :.4f 格式说明符。[功能] 将浮点数强制保留 4 位小数转换为字符串
                f"{metrics_before['recall']:.4f}",
                f"{metrics_before['f1']:.4f}",
                f"{metrics_before['roc_auc']:.4f}",
                f"{metrics_before['pr_auc']:.4f}",
                f"{metrics_before['tp']}",
                f"{metrics_before['fp']}",
                f"{metrics_before['fn']}",
            ],
            '点调整后': [
                f"{metrics_after['precision']:.4f}",
                f"{metrics_after['recall']:.4f}",
                f"{metrics_after['f1']:.4f}",
                f"{metrics_after['roc_auc']:.4f}",
                f"{metrics_after['pr_auc']:.4f}",
                f"{metrics_after['tp']}",
                f"{metrics_after['fp']}",
                f"{metrics_after['fn']}",
            ],
            '改进': [
                f"{metrics_after['precision'] - metrics_before['precision']:+.4f}",  # [语法] :+.4f。[功能] 带显式正负号格式化，直观展示指标提升的绝对增量
                f"{metrics_after['recall'] - metrics_before['recall']:+.4f}",
                f"{metrics_after['f1'] - metrics_before['f1']:+.4f}",
                f"{metrics_after['roc_auc'] - metrics_before['roc_auc']:+.4f}",
                f"{metrics_after['pr_auc'] - metrics_before['pr_auc']:+.4f}",
                f"{metrics_after['tp'] - metrics_before['tp']:+d}",  # [语法] :+d。[功能] 带正负号格式化整数，展示个数的增减
                f"{metrics_after['fp'] - metrics_before['fp']:+d}",
                f"{metrics_after['fn'] - metrics_before['fn']:+d}",
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        print("\n═ 详细评估对比报告 ═\n" + report_df.to_string(index=False))  # [语法] to_string(index=False)。[功能] 以字符流表格化排版控制台输出并剔除丑陋的行索引号
        return report_df

    def _compute_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray, stage: str) -> Dict:
        # [语法] sklearn confusion_matrix 并调用 ravel() 展平一维，多变量解包。[功能] 获取混淆矩阵的四大核心计数：真负、假正、假负、真正
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        
        # [语法] 三元表达式保护除零。[功能] 分别套用机器学习核心公式计算 精确率、召回率、F1调和平均
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # [功能] 若该测试段标签含有异常（防崩溃条件判定），计算 ROC-AUC 及 PR-AUC 两个积分曲线指标
        roc_auc = roc_auc_score(true_labels, self.scores) if len(np.unique(true_labels)) > 1 else 0
        pr_auc = average_precision_score(true_labels, self.scores) if len(np.unique(true_labels)) > 1 else 0
        
        return {
            'precision': precision, 'recall': recall, 'f1': f1,
            'roc_auc': roc_auc, 'pr_auc': pr_auc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }

    def analyze_variable_contribution(self) -> pd.DataFrame:
        """根因追溯机制：找出工业系统中导致全局异常得分突增的罪魁祸首变量"""
        if not hasattr(self, 'norm_var_error'): raise ValueError("Must call compute_anomaly_scores() first")
        print("\n═ 变量贡献度分析 ═")
        
        # [语法] np.argmax 沿第二维度(特征轴)检索索引。[功能] 提取出在每个时刻中产生最大归一化残差的传感器通道序号
        max_var_indices = np.argmax(self.norm_var_error, axis=1)
        
        # [语法] pd.Series 的 value_counts 及 sort_index。[功能] 将这些通道序号进行直方频数统计，并重组为顺序索引序列
        var_contributions = pd.Series(max_var_indices).value_counts().sort_index()
        
        # [语法] 字典构建 DataFrame 加上推导式遍历。[功能] 将各物理变量及对应的触发次数映射为标准表格
        contrib_df = pd.DataFrame({
            'Variable': range(self.num_vars),
            'Anomaly Count': [var_contributions.get(i, 0) for i in range(self.num_vars)],
        })
        # [语法] 列向量广播计算并链式调用 round()。[功能] 计算该变量致错次数在全时间序列中的百分比
        contrib_df['Percentage'] = (contrib_df['Anomaly Count'] / len(self.scores) * 100).round(2)
        
        # [语法] sort_values 降序排列。[功能] 将导致异常最频繁的焦点特征排序在表格头部
        contrib_df = contrib_df.sort_values('Anomaly Count', ascending=False)
        print(contrib_df.to_string(index=False))
        return contrib_df

    def cluster_anomaly_segments(self, pred_labels: np.ndarray) -> List[Tuple[int, int]]:
        """连续异常事件聚类：将离散的异常点位组合并划分为有始有终的实体时间段（连通分量解析）"""
        print("\n═ 异常片段聚类 ═")
        anomaly_indices = np.where(pred_labels == 1)[0]  # [语法] np.where 返回元组的第0个元素。[功能] 提取出所有预测为 1（异常）的绝对索引位置
        
        if len(anomaly_indices) == 0:
            print("  无异常片段")
            return []
        
        segments = []  # [功能] 存放由起止索引元组构成的闭合区间列表
        start = prev = anomaly_indices[0]  # [语法] 多变量平行赋值。[功能] 双指针法初始化起点及前驱节点
        
        for idx in anomaly_indices[1:]:  # [语法] 列表切片跳过头部元素遍历。[功能] 扫描检查相邻索引的连续性
            if idx - prev > 1:  # [语法] 算术判定。[功能] 索引差值大于 1，说明时序链在此处断裂，旧事件已结束，新事件将开始
                segments.append((start, prev + 1))  # [功能] 把左闭右开的刚结束事件压入段列表
                start = idx  # [功能] 重置起点到新的断点处
            prev = idx
        
        segments.append((start, prev + 1))  # [功能] 循环结束后，收尾处理缓存中遗留的最后一个未闭合事件段
        
        for i, (seg_start, seg_end) in enumerate(segments, 1):  # [语法] enumerate 解析附带起始计数值。[功能] 遍历格式化输出
            print(f"  片段 {i}: [{seg_start:5d}, {seg_end:5d}] - 长度 {seg_end - seg_start:3d}")
        return segments


# ==========================================
# 模块七：三段式联动对比图表绘制
# 功能：通过 Matplotlib 可视化渲染得分波动、阈值切割面，以及 PA 调整前后对于攻击底纹的覆盖效力差异
# ==========================================
    def visualize_comparison(self, pred_labels_before: np.ndarray, pred_labels_after: np.ndarray, figsize: Tuple = (18, 10)):
        # [语法] subplots 初始化图表对象与轴数组。[功能] 创建一个 3行1列 尺寸为 18x10，且横坐标时间轴保持联动的全局画板
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        x = np.arange(len(self.scores))  # [语法] np.arange 序列生成。[功能] 构建支撑横轴绘图的全长整数序列（样本索引）
        
        # --- 子图 1：原始分值与切分界线 ---
        axes[0].plot(x, self.scores, color='#1f77b4', linewidth=1.5, label='Anomaly Score')  # [功能] 绘制连绵波动的多维综合异常折线图
        axes[0].axhline(self.threshold, color='#d62728', linestyle='--', linewidth=2, label=f'Threshold={self.threshold:.4f}')  # [功能] 绘制贯穿整个横轴的红虚线（拦截阈值）
        axes[0].set_ylabel('Anomaly Score')
        axes[0].set_title('Anomaly Score')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        
        # --- 子图 2：调整前（生判定区覆盖） ---
        # [语法] fill_between 基于布尔掩码填色。[功能] 将检测为 1 的对应背景染成半透明橙色
        axes[1].fill_between(x, 0, 1, where=(pred_labels_before == 1), alpha=0.3, color='#ff7f0e', label='Detected Anomaly')
        if self.labels_df is not None:
            true_labels = self._extract_labels_for_samples()
            if true_labels is not None:
                # [功能] 叠加一层半透明绿色的绝对真实攻击底纹，观察橙色面积是否命中了绿色面积
                axes[1].fill_between(x, 0, 1, where=(true_labels == 1), alpha=0.2, color='#2ca02c', label='True Anomaly')
        axes[1].set_ylabel('Anomaly Label'); axes[1].set_title(f'Before Point Adjustment (Anomalies: {pred_labels_before.sum()})')
        axes[1].set_ylim(-0.1, 1.1); axes[1].legend()  # [语法] set_ylim 限制视图范围。[功能] 将 Y 轴死锁在 [0, 1] 使得背景填充完全贴顶填底
        
        # --- 子图 3：调整后（熟判定区覆盖） ---
        # [功能] 展示应用 PA 策略向前后左右容错扩张后的判定绿区
        axes[2].fill_between(x, 0, 1, where=(pred_labels_after == 1), alpha=0.3, color='#2ca02c', label='Detected Anomaly (Adjusted)')
        if self.labels_df is not None:
            true_labels = self._extract_labels_for_samples()
            if true_labels is not None:
                # [功能] 同上，将扩充后的预测块叠加在红色真实底纹上进行终极视觉比较
                axes[2].fill_between(x, 0, 1, where=(true_labels == 1), alpha=0.2, color='#d62728', label='True Anomaly')
        axes[2].set_xlabel('Sample Index'); axes[2].set_ylabel('Anomaly Label')
        axes[2].set_title(f'After Point Adjustment (Anomalies: {pred_labels_after.sum()})')
        axes[2].set_ylim(-0.1, 1.1); axes[2].legend()
        
        plt.tight_layout()  # [语法] API调用。[功能] 指示渲染引擎自我计算间距，防止标题等文本标签重叠
        return fig


# ==========================================
# 模块八：磁盘 IO 落盘归档
# 功能：将计算完成的得分流、两种标签流以及综合比对指标序列存至本地作为工程复用基底
# ==========================================
    def save_results(self, output_prefix: str = 'advanced_'):
        print(f"\n═ 保存结果 ═")
        
        # [语法] Path 除法运算符重载。[功能] 拼接生成本地硬盘文件的目标全路径
        score_path = self.save_folder / f'{output_prefix}anomaly_score.npy'
        label_before_path = self.save_folder / f'{output_prefix}anomaly_label_before_adjustment.npy'
        label_after_path = self.save_folder / f'{output_prefix}anomaly_label_after_adjustment.npy'
        
        # [语法] np.save 高效读写流。[功能] 以原生 numpy 二进制格式固化一维数据（比 CSV 更小更快）
        np.save(score_path, self.scores)
        np.save(label_before_path, self.pred_labels)
        np.save(label_after_path, self.pred_labels_adjusted)
        
        csv_path = self.save_folder / f'{output_prefix}anomaly_results.csv'
        result_df = pd.DataFrame({  # [功能] 构建汇总宽表对齐并写入所有分析列
            'sample_idx': np.arange(len(self.scores)),
            'anomaly_score': self.scores,
            'label_before': self.pred_labels,
            'label_after': self.pred_labels_adjusted,
        })
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # [语法] 存入指定编码的CSV。[功能] 阻止默认写入无效的前置递增整数列，利用 sig 解决 Windows Excel 乱码
        
        print(f"  ✓ {score_path}\n  ✓ {label_before_path}\n  ✓ {label_after_path}\n  ✓ {csv_path}")

    def run_pipeline(self, threshold_method: str = 'auto', output_prefix: str = 'advanced_'):
        """全局统筹串联调度主轴（门面方法）"""
        print("\n" + "="*60 + "\n异常检测后处理管道\n" + "="*60)
        
        self.compute_anomaly_scores()  # [功能阶段 1] 求取特征残差及标定绝对得分
        self.select_threshold(method=threshold_method)  # [功能阶段 2] 确立决策切割面
        self.pred_labels = (self.scores > self.threshold).astype(int)  # [功能阶段 3] 硬划分原始判断标签
        self.pred_labels_adjusted = self.apply_point_adjustment(self.pred_labels)  # [功能阶段 4] 实施局部时域容错扩充
        
        self.analyze_variable_contribution()  # [功能阶段 5] 构建主要错误贡献表
        
        print("\n点调整前：")
        segments_before = self.cluster_anomaly_segments(self.pred_labels)  # [功能阶段 6a] 获取调整前的散碎事件块段落
        print("\n点调整后：")
        segments_after = self.cluster_anomaly_segments(self.pred_labels_adjusted)  # [功能阶段 6b] 获取调整后聚集成型的事件块段落
        
        report = self.generate_evaluation_report(self.pred_labels, self.pred_labels_adjusted)  # [功能阶段 7] 比照真值计算性能量化收益表
        fig = self.visualize_comparison(self.pred_labels, self.pred_labels_adjusted)  # [功能阶段 8] 调用底层引擎绘制分屏对比大图
        self.save_results(output_prefix=output_prefix)  # [功能阶段 9] 最终结果下盘固化
        
        return {  # [语法] 返回一个大型复合字典。[功能] 提供灵活的编程 API，以便在调用端抽取任何想要复用的流程数据或报表
            'scores': self.scores, 'threshold': self.threshold,
            'labels_before': self.pred_labels, 'labels_after': self.pred_labels_adjusted,
            'segments_before': segments_before, 'segments_after': segments_after,
            'report': report, 'figure': fig,
        }


# ==========================================
# 模块九：应用入口与用例演示
# 功能：脚本执行的标准接驳口
# ==========================================
def main():
    import sys  # [语法] 导入解释器环境模块。[功能] 常用于读取命令行 args
    
    # [语法] 使用 r 前缀的原始字符串。[功能] 阻止 Windows 系统路径中的反斜杠被识别为转义符（例如防止 \Test 变成制表符）
    save_folder = Path(r'E:\PyDeepLearningLab\Code\07_RA-STGNN\Test')
    label_path = Path(r'E:\PyDeepLearningLab\Code\07_RA-STGNN\Data\WADI\data_10s.pkl')
    
    pipeline = AnomalyDetectionPipeline(save_folder, label_path=label_path)  # [功能] 实例化刚定义好的工具架子
    results = pipeline.run_pipeline(threshold_method='auto')  # [功能] 以 auto 有监督参数拉起整套计算流并捕获结果载荷
    
    plt.show()  # [语法] 调用阻塞事件循环。[功能] 阻断后台程序退出，强行弹窗挂起那张生成好的对比分析图

if __name__ == '__main__':  # [语法] 主程序入口守卫。[功能] 防止这段代码在被当作库 import 到别处时遭到意外直接运行
    main()