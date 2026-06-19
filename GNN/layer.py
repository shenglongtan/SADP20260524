from __future__ import division  # 从未来的Python版本导入精确除法，确保在Python 2环境中 '/' 也是浮点除法（向下兼容）
import torch  # 导入 PyTorch 核心计算库，提供多维张量(Tensor)数据结构和数学运算
import torch.nn as nn  # 导入 PyTorch 神经网络模块，包含构建网络所需的层（如 Conv2d, Linear, Parameter 等基类）
from torch.nn import init  # 导入 PyTorch 的参数初始化模块，用于对网络权重进行特定分布的初始化（如全1、全0等）
import numbers  # 导入 Python 内置的数字抽象基类模块，后续用于类型判断（如判断变量是否为整数）
import torch.nn.functional as F  # 导入 PyTorch 的函数式接口，包含激活函数（ReLU, Tanh）、归一化等无内部状态的函数
class nconv(nn.Module):
    """
    基础节点图卷积层 (Node Convolution)。
    该类使用外部传入的静态/固定邻接矩阵 A 对输入特征 x 执行空间拓扑聚合。
    """
    def __init__(self):
        # 调用父类 nn.Module 的初始化函数，注册该模块
        super(nconv, self).__init__()
    def forward(self, x, A):
        # 前向传播逻辑：使用爱因斯坦求和约定 (einsum) 高效执行多维张量乘法。
        # 'ncwl' 代表输入特征 x 的维度：n(批量 batch_size), c(通道 channels), w(源节点 source_nodes), l(时间步序列长度 seq_length)。
        # 'vw' 代表邻接矩阵 A 的维度：v(目标节点 target_nodes), w(源节点 source_nodes)。
        # '->ncvl' 表示输出维度：保持 n, c, l 不变，将 w 维度（源节点）与邻接矩阵相乘求和，映射到 v 维度（目标节点）上。
        # 物理意义：中心节点 v 根据邻接矩阵 A 的权重，吸收周边邻居节点 w 的特征。
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        # .contiguous() 确保经过底层维度重排后的张量在物理内存中是连续分布的，避免后续操作（如 view, reshape）报错
        return x.contiguous()
class dy_nconv(nn.Module):
    """
    动态节点图卷积层 (Dynamic Node Convolution)。
    与上面的 nconv 不同，这里使用的邻接矩阵 A 是一个四维张量，它随批次(batch)和时间步(length)动态变化。
    """
    def __init__(self):
        # 调用父类 nn.Module 的初始化函数
        super(dy_nconv, self).__init__()
    def forward(self, x, A):
        # 'ncvl' 代表输入特征 x：n(batch), c(channels), v(源节点), l(length)。
        # 'nvwl' 代表动态邻接矩阵 A：n(batch), v(源节点), w(目标节点), l(length)。
        # '->ncwl' 表示在每个批次 n 和时间步 l 独立计算，将通道 c 的特征从源节点 v 映射到目标节点 w。
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        # 同样保证内存连续性并返回结果
        return x.contiguous()
class linear(nn.Module):
    """
    基于 1x1 二维卷积实现的特征维度线性映射层。
    在处理 [batch, channels, nodes, length] 的四维张量时，使用 1x1 卷积代替传统的 nn.Linear，
    可以避免把张量展平(flatten)再恢复的繁琐操作，直接在通道(channels)维度上进行线性组合。
    """
    def __init__(self, c_in, c_out, bias=True):
        # c_in: 输入通道数; c_out: 输出通道数; bias: 是否启用偏置项
        super(linear, self).__init__()
        # 定义 2D 卷积层：核大小为 1x1，步长为 1，不进行边缘填充(padding=0)。
        # 这种设置相当于对每个节点在每个时间步上，独立施加一个全连接层（仅改变通道数量）。
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)
    def forward(self, x):
        # 将输入 x 送入 1x1 卷积层执行前向计算并返回
        return self.mlp(x)
class prop(nn.Module):
    """
    标准图传播层 (Propagation Layer)。
    沿着图拓扑结构执行多步（跳数）的图卷积传播，并将最后一步的特征通过线性层映射输出。
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        # gdep: 图传播的深度/最大跳数; dropout: 随机失活率(此处未实际使用); alpha: 根节点特征保留比例
        super(prop, self).__init__()
        # 实例化基础的节点卷积算子 nconv
        self.nconv = nconv()
        # 实例化自定义的 1x1 卷积线性映射层
        self.mlp = linear(c_in, c_out)
        # 保存最大跳数
        self.gdep = gdep
        # 保存 dropout 概率
        self.dropout = dropout
        # 保存信息保留系数
        self.alpha = alpha
    def forward(self, x, adj):
        # adj.size(0) 获取节点总数，torch.eye 生成对角线为1的单位矩阵，.to(x.device) 将其放到与 x 相同的设备（CPU/GPU）上。
        # 这一步是为邻接矩阵添加自环（Self-loop），确保节点在聚合时包含自身的初始信息。
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        # 沿着维度1（即每一行）对邻接矩阵求和，得到每个节点的出度（Degree）向量
        d = adj.sum(1)
        # 初始化隐藏状态 h 为初始输入 x（0跳特征）
        h = x
        # 将度数向量赋给变量 dv
        dv = d
        # dv.view(-1, 1) 将一维度数向量转为列向量，利用广播机制让邻接矩阵除以对应节点的度数，实现马尔可夫行归一化 (A_norm = D^-1 * A)
        a = adj / dv.view(-1, 1)
        # 根据设定的传播深度 gdep，循环执行图卷积
        for i in range(self.gdep):
            # 核心防过平滑机制：每一跳的新特征 h，由一定比例的初始输入 (alpha * x) 和聚合邻居的新特征 ((1 - alpha) * nconv) 组合而成
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        # 循环结束后，将最后一跳的隐藏状态 h 传入线性层进行通道映射
        ho = self.mlp(h)
        # 返回最终的传播结果
        return ho
class mixprop(nn.Module):
    """
    混合跳数传播层 (Mix-hop propagation layer)。
    与上方 prop 层的区别在于：prop 只取最后一步的结果，而 mixprop 会将第 0 跳到第 gdep 跳的所有特征在通道维度上拼接起来，
    统一进行映射，从而同时捕捉局部与全局的空间依赖。
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        # 参数初始化与 prop 层一致
        super(mixprop, self).__init__()
        # 实例化基础图卷积算子
        self.nconv = nconv()
        # 注意此处的输入通道数变为 (gdep + 1) * c_in，因为 0 到 gdep 一共包含了 gdep+1 个跳数的特征矩阵拼接在一起
        self.mlp = linear((gdep + 1) * c_in, c_out)
        # 保存传播深度
        self.gdep = gdep
        # 保存 dropout 概率
        self.dropout = dropout
        # 保存保留系数
        self.alpha = alpha
    def forward(self, x, adj):
        # 为邻接矩阵增加自环（对角线加 1）
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        # 计算每个节点的度数向量
        d = adj.sum(1)
        # 初始状态设为输入特征 x
        h = x
        # 建立一个列表 out，用于收集每一跳的特征，首先装入第 0 跳（未传播）的原始特征
        out = [h]
        # 对邻接矩阵进行行归一化
        a = adj / d.view(-1, 1)
        # 开始循环传播，共进行 gdep 次
        for i in range(self.gdep):
            # 每一步同样保留一定比例的初始输入特征 (alpha * x)，防止网络加深导致所有节点特征趋同（过平滑问题）
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            # 将第 i+1 跳的特征追加到列表中
            out.append(h)
        # 使用 torch.cat 将列表中收集的 gdep+1 个四维张量，沿着维度 1（即通道 channels 维度）拼接起来
        ho = torch.cat(out, dim=1)
        # 将拼接后的“超厚”特征图传入 1x1 卷积线性层，降维到目标输出通道数 c_out
        ho = self.mlp(ho)
        # 返回融合了多跳邻居信息的最终特征
        return ho
class dy_mixprop(nn.Module):
    """
    动态混合跳数传播层 (Dynamic Mix-hop propagation layer)。
    不使用外部传入的先验图，而是基于输入特征的隐式映射实时计算动态图，然后在其上执行双向的混合跳数传播。
    """
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        # 实例化支持四维动态矩阵的图卷积算子 dy_nconv
        self.nconv = dy_nconv()
        # mlp1 用于处理前向动态图传播特征的降维映射
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        # mlp2 用于处理反向动态图传播特征的降维映射
        self.mlp2 = linear((gdep + 1) * c_in, c_out)
        # 保存超参数
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        # lin1 和 lin2 是用于分别生成“源节点查询向量”和“目标节点键向量”的 1x1 卷积层
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)
    def forward(self, x):
        # 提取用于构建动态图的源节点特征表示，并使用 Tanh 激活将其值限制在 [-1, 1]
        x1 = torch.tanh(self.lin1(x))
        # 提取用于构建动态图的目标节点特征表示，同样使用 Tanh 激活
        x2 = torch.tanh(self.lin2(x))
        # x1.transpose(2, 1) 交换节点和通道维度，利用 dy_nconv 执行类似于注意力机制的内积操作，计算节点对之间的动态关系分数 adj
        adj = self.nconv(x1.transpose(2, 1), x2)
        # 沿着目标节点维度 (dim=2) 进行 Softmax 操作，使矩阵每一行的概率和为 1，生成前向动态转移概率矩阵 adj0
        adj0 = torch.softmax(adj, dim=2)
        # 将原始得分矩阵在节点维度进行转置（交换源和目标），再执行 Softmax，生成反向动态转移概率矩阵 adj1
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)
        # ====== 针对 adj0 (前向图) 执行混合跳数传播 ======
        h = x  # 初始化隐藏状态
        out = [h]  # 收集 0 跳特征
        for i in range(self.gdep):  # 循环 gdep 跳
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)  # 使用动态算子 dy_nconv 在前向图上传播
            out.append(h)  # 收集第 i+1 跳特征
        ho = torch.cat(out, dim=1)  # 拼接通道
        ho1 = self.mlp1(ho)  # 降维映射，得到前向传播的最终特征 ho1
        # ====== 针对 adj1 (反向图) 执行混合跳数传播 ======
        h = x  # 重新初始化隐藏状态为原始输入
        out = [h]  # 收集 0 跳特征
        for i in range(self.gdep):  # 循环 gdep 跳
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)  # 在反向图上进行传播
            out.append(h)  # 收集特征
        ho = torch.cat(out, dim=1)  # 拼接通道
        ho2 = self.mlp2(ho)  # 降维映射，得到反向传播的最终特征 ho2
        # 将前向和反向传播提取到的特征直接相加融合，作为本层的最终输出
        return ho1 + ho2
class dilated_1D(nn.Module):
    """
    1维扩张卷积层（冗余测试类，核心逻辑在下面的 dilated_inception 中实现）。
    用于在时间轴上提取序列的长距离依赖特征。
    """
    def __init__(self, cin, cout, dilation_factor=2):
        # cin: 输入通道; cout: 输出通道; dilation_factor: 扩张步长
        super(dilated_1D, self).__init__()
        # 以下两行代码声明了未使用的数据结构，可能是开发过程中的遗留代码
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        # 覆盖上方的 self.tconv，定义一个标准的 2D 卷积层：
        # 核大小 (1, 7) 意味着不在空间（节点）维度上跨越，只在时间维度上跨越 7 步。
        # dilation=(1, dilation_factor) 让时间维度的卷积核具有膨胀间隔，从而扩大感受野。
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))
    def forward(self, input):
        # 直接让输入通过带有膨胀因子的二维卷积层
        x = self.tconv(input)
        return x
class dilated_inception(nn.Module):
    """
    扩张 Inception 模块 (Dilated Inception Module)。
    通过使用多个不同时间核大小的并行扩张卷积，能够同时捕捉多种时间尺度（短、中、长）的序列动态特征。
    """
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        # 创建一个 ModuleList 容器，用于存放并行的各个卷积分支
        self.tconv = nn.ModuleList()
        # 定义 4 种不同的时间维度卷积核大小
        self.kernel_set = [2, 3, 6, 7]
        # 为了保证拼接后的总输出通道数等于期望的 cout，将 cout 均分给 4 个不同的核分支
        cout = int(cout / len(self.kernel_set))
        # 遍历每一种核大小，创建一个对应的时间扩张卷积层，并添加到模块列表中
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
    def forward(self, input):
        # 创建一个空列表，用于收集各个卷积分支的前向传播结果
        x = []
        for i in range(len(self.kernel_set)):
            # 依次将输入传入不同的卷积层，并将输出张量追加到列表 x 中
            x.append(self.tconv[i](input))
        # 关键的时间维度对齐操作：
        # 因为不同分支的卷积核大小不一样，经过不带 Padding 的卷积后，导致产生的时间序列长度也会参差不齐。
        # 卷积核越长（最大为7），卷积后剩下的有效时间步越短（x[-1] 的时间维度最短）。
        for i in range(len(self.kernel_set)):
            # x[-1].size(3) 获取时间维度的最小长度。
            # x[i][..., -x[-1].size(3):] 表示在倒数第一个维度（时间维度）上截取末尾相同长度的数据段。
            # 物理意义：舍弃掉大感受野无法触及的早期历史，只保留各分支最近的相同长度的时间段。
            x[i] = x[i][..., -x[-1].size(3):]
        # 使用 torch.cat 将这 4 个在时间长度上对齐的张量，沿着通道维度 (dim=1) 进行无缝拼接
        x = torch.cat(x, dim=1)
        # 返回融合了多尺度时间特征的特征图
        return x
class GraphConstructor(nn.Module):
    """
    全局自适应图构造器 (Graph Constructor)。
    利用节点的静态特征或随机初始化的可学习嵌入（Embedding），基于两者的相似度来推断全局的拓扑邻接关系，
    解决工业场景下难以获取真实连通图（先验知识）的问题。
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        # 保存节点的数量
        self.nnodes = nnodes
        if static_feat is not None:
            # 如果系统提供了先验的静态特征（如经纬度、传感器类型编码）
            xd = static_feat.shape[1]  # 获取静态特征的维度大小
            # 构造两个线性层，用于分别从静态特征映射出源节点表示(lin1)和目标节点表示(lin2)
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            # 如果没有静态特征，则完全依靠反向传播来学习节点的隐式特征
            # 定义源节点的独立嵌入层（对应论文中的 E1）
            self.emb1 = nn.Embedding(nnodes, dim)
            # 定义目标节点的独立嵌入层（对应论文中的 E2）
            self.emb2 = nn.Embedding(nnodes, dim)
            # 对应的两个线性映射层
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        # 保存运行设备，如 'cuda'
        self.device = device
        # 保存 k 值，代表最终每个节点只保留关联度最强的 top-k 个邻居
        self.k = k
        # 隐藏空间维度
        self.dim = dim
        # 控制 Tanh 函数饱和程度的缩放因子
        self.alpha = alpha
        # 将静态特征保存为类的属性
        self.static_feat = static_feat
    def forward(self, idx):
        # idx 是当前需要推断的节点索引张量
        if self.static_feat is None:
            # 根据索引，从字典中查表获取源节点和目标节点的嵌入向量
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            # 直接通过索引切片获取静态特征矩阵中对应的数据
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1  # 均复用同一个静态特征矩阵
        # 线性映射后，乘以 alpha 调整梯度区域，再应用 tanh 将向量值域严格约束在 [-1, 1] 之间
        # 分别生成 M1 和 M2
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        # 计算关系得分 a，执行非对称相减操作： M1 * M2^T - M2 * M1^T
        # 这种交叉相减机制确保了生成的有向图不至于完全退化，并使得关系度量更为复杂鲁棒
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        # 乘以 alpha 并进行 tanh 激活，最后通过 F.relu 截断所有的负数得分（负关联意味着物理断连，置为0）
        adj = F.relu(torch.tanh(self.alpha * a))
        # ====== 下方为 Top-k 稀疏化过程 ======
        # 创建一个与 adj 维度相同的全 0 矩阵作为掩码底板，并发送到相应设备
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))  # 显式全部置0
        # 对当前的邻接矩阵加上一个非常微弱的随机扰动（0.01量级），然后沿行方向获取前 K 个最大值 (s1) 及其列索引 (t1)。
        # 加扰动的目的是防止存在大量相同的权重值（如0）时 topk 操作选择不稳定导致梯度计算困扰。
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        # 根据返回的列索引 t1，将掩码矩阵 mask 对应位置填充为 1
        mask.scatter_(1, t1, s1.fill_(1))
        # 使用 mask 过滤原始邻接矩阵，使得不是 Top-k 的连接全变为 0，生成稀疏图
        adj = adj * mask
        return adj
    def fullA(self, idx):
        """
        全连接图推断模式。
        返回未经 Top-K 掩码强制清零处理的完整（稠密）关系得分矩阵。
        """
        # 前序的向量获取和投影逻辑与 forward 方法完全一致
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        # 不生成 mask，直接返回完整的非负矩阵
        return adj
class graph_global(nn.Module):
    """
    最基础直接的全局图构造器。
    它不依赖任何隐式特征的点积计算，而是直接声明一个 N x N 大小的参数矩阵，完全交由反向传播（梯度）来学习所有的连接权重。
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        # 直接使用 torch.randn 随机初始化一个全尺寸的权重矩阵，并使用 nn.Parameter 包装，告知 PyTorch 这是一个需要优化更新的参数
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)
    def forward(self, idx):
        # 忽略传入的索引，直接对整个参数矩阵使用 ReLU 进行负值截断后返回
        return F.relu(self.A)
class graph_undirected(nn.Module):
    """
    无向图构造器。
    通过强制源节点和目标节点共享同一个嵌入字典（emb1），计算出来的内积矩阵天然具有对称性，从而生成物理结构上的无向图。
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            # 相比 GraphConstructor，这里只保留了一组 emb1 和 lin1
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx):
        if self.static_feat is None:
            # 无论是源还是目标，都从 emb1 查表取值
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        # 使用同样的 lin1 进行映射
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))
        # 计算内积（由于两个矩阵相同，M1 * M1^T 必然是对称矩阵）
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        # 激活去负数
        adj = F.relu(torch.tanh(self.alpha * a))
        # ====== Top-k 掩码过滤 ======
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)  # 获取最大的K个值及其坐标
        mask.scatter_(1, t1, s1.fill_(1))  # 根据坐标将掩码置1
        adj = adj * mask  # 相乘过滤
        return adj
class graph_directed(nn.Module):
    """
    常规的有向图构造器变体。
    它使用了独立的源和目标嵌入，但不包含复杂交叉相减机制，仅通过单纯的非对称内积生成图结构。
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            # 存在独立的两组嵌入与映射网络
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        # 映射激活
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        # 直接进行内积计算，M1 * M2^T，由于 M1 和 M2 通常不同，生成天然的非对称有向邻接矩阵
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        # 激活去负数
        adj = F.relu(torch.tanh(self.alpha * a))
        # ====== Top-k 掩码过滤 ======
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj
class LayerNorm(nn.Module):
    """
    自定义的层归一化 (Layer Normalization)。
    这个版本专为节点可切分的图网络设计：在进行数据并行或子图训练(Split Batch)时，
    通过传入子图的节点索引 idx，使得网络只提取并在局部节点上应用仿射变换(scale & shift)参数，极大节约显存计算开销。
    """
    # 提前声明常量成员变量列表，以便 PyTorch 的 TorchScript 进行底层 JIT 优化编译
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        @param normalized_shape: 期望进行归一化的所有维度构成的元组（对于STGNN通常是 [channels, nodes, seq_length]）。
        @param eps: 加在方差上的极小常数值，用于防止标准化除零崩溃。
        @param elementwise_affine: 是否保留并学习特征变换参数 (缩放项 weight 和 平移项 bias)。
        """
        super(LayerNorm, self).__init__()
        # 判断传入的形状如果只是单一整数，将其转换为元组格式以兼容后续操作
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # 存储标准化形状
        self.normalized_shape = tuple(normalized_shape)
        # 存储极小值
        self.eps = eps
        # 存储布尔标志，记录是否需要仿射参数
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # 声明全局维度的 weight 为模型可学习参数 Parameter，并按 normalized_shape 构建张量空间
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            # 同理，声明全局的 bias
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            # 如果不使用仿射参数，安全地将它们在模块的参数注册表中标记为 None
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        # 触发权重初始化方法
        self.reset_parameters()
    def reset_parameters(self):
        """
        初始化或重置仿射变换参数。
        将作为乘法系数的 weight 统一置为 1，将作为加法偏移的 bias 统一置为 0。
        """
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def forward(self, input, idx):
        """
        前向传播计算。
        @param input: 输入特征张量，格式通常为 [batch, channel, node, length]。
        @param idx: 需要当前批次处理的节点索引集合。
        """
        if self.elementwise_affine:
            # tuple(input.shape[1:]) 提取了输入张量除 batch 外的后面三个维度用于作为归一化的参照尺寸。
            # 这里是最核心的显存节约技巧：self.weight[:, idx, :] 表示对于第二个维度（节点维度），
            # 只抽出列表中 idx 存在的局部节点对应的那几块参数进入底层的 F.layer_norm 运算。
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            # 若不需要仿射参数，直接调用标准的 F.layer_norm
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)
    def extra_repr(self):
        """
        魔术方法：使得开发者在终端使用 print(model) 打印整个网络架构时，
        该归一化层旁边能直观显示出自定义的参数信息(如形状、eps等)，便于排错检查。
        """
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)