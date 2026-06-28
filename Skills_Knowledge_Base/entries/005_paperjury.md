# PaperJury - 论文投稿前 AI 评审压力测试

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-005                                                           |
| 来源     | https://github.com/u7079256/paperjury                              |
| 收录日期 | 2026-06-28                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #claude-code #论文评审 #预提交 #LaTeX #对抗式评审 #闭环修订  |
| 重要程度 | ★★★★★                                                             |
| 版本     | v1.2.1 (2026-06-15)                                               |
| 许可证   | MIT                                                                |
| 热度     | 462 stars / 33 forks                                               |
| 兼容性   | Claude Code - 可安装 ✅                                             |

## 核心摘要

> 论文投稿前的 AI 评审压力测试工具。采用闭环工作流：评审（review）→ 裁决（verdict）→ 修订（revise）→ 验证（verify）。将反馈分为三类：安全文本修复、需要作者判断的项目、无效批评。包含 11 个确定性组件和 10 个语义组件，支持 LaTeX/Markdown/Word 格式，强调"不能替代作者科学判断或同行评审"。

## 关键知识点

- **闭环工作流**：review → verdict → revise → verify，每轮修改都有验证
- **三类反馈分级**：✅ 安全修复（可直接改）/ 🧑‍💻 作者处理（需人类判断）/ 🛑 无效（AI 误读）
- **三种操作模式**：direct-edit（定向修改）/ review（完整评审）/ auto（多轮迭代）
- **11 个确定性组件**：文档分解、段落编号、补丁原子化应用与回滚、LaTeX 编译验证、投稿合规筛查等
- **10 个语义组件**：评审员分配、覆盖度审计、庭审式辩论、润色、召回审计等
- **安全护栏**：6 条核心规则，未经作者同意不得修改、评审员相互隔离、编辑日志只追加不删除
- **支持格式**：LaTeX (.tex) / Markdown (.md) / Word (.docx)
- **路线图**：计划支持会议特定评审人设模板（CVPR、ACL、NeurIPS）

## 详细笔记

### 三种操作模式

| 模式 | 场景 | 行为 |
|------|------|------|
| direct-edit | 修改特定段落 | 跳过评审面板，直接起草补丁，需作者批准 |
| review | 完整评审/模拟审稿 | 激活对抗式评审引擎，逐项确认修改 |
| auto | 多轮迭代 | 需配置 `/goal`，自动应用安全修复，标记需判断项 |

### 反馈三分类

| 类型 | 说明 | 处理方式 |
|------|------|---------|
| ✅ Safe Fixes | 清晰度、弱声明、结构问题—无需实验 | 可直接应用 |
| 🧑‍💻 Author Processing | 缺失实验、数据、证据 | 需人类判断 |
| 🛑 Invalid | AI 误读论文或建议不当修改 | 丢弃 |

### 11 个确定性组件

1. 文档分解 + 稳定段落编号
2. .docx → Markdown 转换（保留内容）
3. 核心声明提取（仅 auto 模式）
4. 跨会话机器可读账本
5. 只追加的编辑日志
6. 原子化补丁应用 + 回滚能力
7. 冻结声明锚点追踪
8. 交叉引用影响检测
9. 编辑后段落重对齐
10. LaTeX 编译验证（或降级为结构检查）
11. 投稿合规筛查

### 10 个语义组件

1. 评审员分配
2. 整体阅读检查
3. 覆盖度审计
4. 去重
5. 庭审式辩论
6. 润色通道
7. 召回审计
8. 起草
9. 编辑审计
10. 收敛判定

### 6 条安全规则

1. 未经作者明确批准不得修改
2. 评审员/陪审员相互隔离
3. 每个可修复问题有明确纠正标准
4. 内部记录不计入被评审文本
5. 分歧讨论决定，人类覆盖有记录
6. 运行时路径解析（无硬编码配置）

### 仓库结构

```
paperjury/
├── .claude-plugin/        # Marketplace 打包
├── workflows/             # 语义阶段
│   ├── assignment
│   ├── reading-check
│   ├── coverage-auditor
│   ├── trial
│   ├── recall-audit
│   └── drafting
├── scripts/               # 确定性检查
│   ├── ledger
│   ├── patch
│   ├── compilation
│   └── compliance
├── references/            # 协议文档
│   ├── review-engine-v3.md
│   ├── ledger-schema.md
│   ├── reviewer-personas
│   └── writing-toolkit
├── docs/                  # 设计文档
│   ├── AGENT-GUIDE.md
│   ├── REVIEW_ENGINE_V3_DESIGN.md
│   └── overview (interactive)
├── samples/dogfood/       # 真实 before/after PDF + 验证报告
└── tests/                 # 脚本和状态机测试
```

## 代码/配置片段

### 安装（Marketplace 推荐）

```
/plugin marketplace add u7079256/paperjury
/plugin install paperjury@u7079256
```

### 手动安装

```bash
# macOS/Linux
git clone https://github.com/u7079256/paperjury ~/.claude/skills/paperjury

# Windows
git clone https://github.com/u7079256/paperjury "$env:USERPROFILE\.claude\skills\paperjury"
```

### 安装验证

```bash
npm run doctor
```

### 使用示例

```
# 快速评审
审稿，重点看实验和 claim 是否站得住。

# 定向修改
把 introduction 这段改紧一些，但不要改变 claim。
```

### 依赖

- **必需**：Node.js
- **可选**：LaTeX 工具链（启用真实编译验证；无则降级为结构检查）
- **更新检查**：禁用 `PAPERJURY_DISABLE_UPDATE_CHECK=1`

## 实践建议

1. **投稿前必用**：论文投稿前跑一轮 PaperJury，能提前发现审稿人会挑的问题
2. **三分类反馈设计精巧**：区分"可直接改"和"需人类判断"的问题，避免 AI 盲目修改实验声明
3. **闭环验证是亮点**：修改后自动验证，不是改完就结束，确保修改不引入新问题
4. **与 SKB-002/004 形成完整链路**：SKB-002 做研究和写作 → SKB-004 提升写作质量 → SKB-005 投稿前压力测试
5. **LaTeX 用户体验最佳**：有 LaTeX 工具链时可以做真实编译验证，建议安装完整 LaTeX 环境
6. **关注路线图**：计划支持 CVPR/ACL/NeurIPS 特定评审人设模板，对顶会投稿非常有用

## 安装状态

- **兼容性**：Claude Code 原生兼容 ✅
- **依赖**：Node.js（必需），LaTeX（可选）
- **当前环境安装**：受网络限制，需在本地执行安装命令

## 关联条目

- [SKB-002 Academic Research Skills](002_academic_research_skills.md) - 全流程编排（含评审环节），本技能专注投稿前深度压力测试
- [SKB-004 Research Paper Writing Skills](004_research_paper_writing_skills.md) - 写作方法论指导，与本技能形成"写作→测试"链路
