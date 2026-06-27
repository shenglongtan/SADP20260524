# SciPilot Figure Skill - 科学论文数据可视化技能

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-003                                                           |
| 来源     | https://github.com/Haojae/scipilot-figure-skill                   |
| 收录日期 | 2026-06-27                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #claude-code #数据可视化 #科学图表 #matplotlib #论文 #Python |
| 重要程度 | ★★★★★                                                             |
| 版本     | v2.1.0                                                            |
| 许可证   | MIT                                                                |
| 热度     | 707 stars / 27 forks                                               |
| 兼容性   | Claude Code - 可安装 ✅                                             |

## 核心摘要

> "先思考，再画图"（thinks first, plots second）。这是一个面向科研人员的 Claude Code 技能，用于生成出版级科学图表。不是简单地渲染图表，而是先做数据画像（profiling），推荐合适的可视化类型，然后执行高质量渲染，最后进行视觉自检。内置多种防错拦截机制，支持中文排版。

## 关键知识点

- **8 步工作流**：理解需求 → 数据画像 → 选择图表类型 → 查询期刊规范 → 配置环境/样式 → 执行渲染 → 视觉自检 → 多格式导出
- **主动防错拦截**：拒绝 n<10 的纯均值柱状图、阻止双 Y 轴欺骗、拒绝饼图/3D 图、捕捉 CJK 字体渲染失败
- **视觉自检循环（v2.1）**：程序审计（检测缺字、文字裁切、标签重叠）+ AI 图像阅读（验证图例遮挡、面板对齐、灰度区分度）
- **中文支持**：`setup_style(lang='zh')`，字体优先级 Noto Sans CJK SC > Source Han Sans SC > SimHei > Microsoft YaHei
- **5 大核心原则**：最终尺寸渲染不缩放、优先矢量格式（PDF/SVG/EPS）、色盲安全调色板、可读排版（7-9pt）、显式标注误差线
- **SciPilot 技能家族**：这是系列中的一个，还有 cite/writing/review/submit/read 等相关技能

## 详细笔记

### 8 步工作流详解

| 步骤 | 说明 |
|------|------|
| 1. 理解需求 | 明确任务目标 |
| 2. 数据画像 | 分析数据类型、样本量、分布、离群值、相关性 |
| 3. 选图类型 | 基于数据特征推荐最合适的图表类型 |
| 4. 期刊规范 | 查询目标期刊的图表格式要求 |
| 5. 配置样式 | 设置环境和视觉风格 |
| 6. 执行渲染 | 生成图表 |
| 7. 视觉自检 | 程序审计 + AI 图像阅读双重验证 |
| 8. 多格式导出 | PDF/SVG/EPS/PNG 等 |

### 主动防错机制

| 拦截场景 | 处理方式 |
|----------|---------|
| n<10 的纯均值柱状图 | 拒绝，推荐 stripplot |
| 双 Y 轴图 | 阻止，防止视觉欺骗 |
| 饼图/3D 图 | 拒绝，建议水平柱状图 |
| CJK 字体渲染失败 | 捕捉并提示修复 |
| 面板字母未对齐 | 自动标注 `layout_tools.add_panel_labels()` |

### 仓库结构

```
scipilot-figure-skill/
├── references/
│   ├── visual_review.md      # 自检审计流程
│   ├── viz_pitfalls.md        # 15 种常见可视化错误及纠正
│   ├── chart_selection.md     # 图表类型选择决策框架
│   ├── journal_specs.md       # 期刊特定格式要求
│   └── plot_recipes.md        # 9 大标准图表类型配方
├── scripts/
│   ├── profile_data.py        # 数据画像
│   ├── setup_style.py         # 样式配置
│   ├── export_figure.py       # 导出
│   └── check_figure.py        # 合规审计
├── SKILL.md                   # 技能入口
├── requirements.txt
└── LICENSE (MIT)
```

### SciPilot 技能家族

| 技能 | 版本 | 用途 |
|------|------|------|
| scipilot-cite-skill | v1.0.0 | 文献发现 |
| **scipilot-figure-skill** | **v2.1.0** | **可视化顾问** |
| scipilot-writing-skill | v1.0.0 | 学术写作 |
| scipilot-review-skill | 计划中 | 同行评审模拟 |
| scipilot-submit-skill | 计划中 | 格式适配 |
| scipilot-read-skill | 计划中 | 论文分析 |

### 技术栈

| 组件 | 技术 |
|------|------|
| 静态渲染 | matplotlib + seaborn + SciencePlots |
| 交互输出 | Plotly |
| 字体处理 | 自动 CJK 配置 + 回退链 |
| 图像验证 | 程序布局审计 + 多模态 AI 审查 |

## 代码/配置片段

### 安装（推荐）

```
Please install this Skill: https://github.com/Haojae/scipilot-figure-skill.git
```

### 手动安装

```bash
git clone https://github.com/Haojae/scipilot-figure-skill.git \
          ~/.claude/skills/scipilot-figure-skill
pip install -r requirements.txt
```

### 依赖

**核心：**
```
matplotlib>=3.7
seaborn>=0.13
plotly>=5.18
pandas>=2.0
numpy>=1.24
scipy>=1.10
Pillow>=10.0
```

**可选增强：**
```
SciencePlots>=2.1
pypdf>=4.0
kaleido>=0.2.1
PyMuPDF>=1.23
```

### 命令行用法

```bash
# 数据画像
python scripts/profile_data.py results.csv --group group --group condition

# 列出 CJK 字体
python scripts/setup_style.py --list-fonts

# 导出演示
python scripts/export_figure.py demo --out ./test_demo

# 合规审计
python scripts/check_figure.py figs/*.pdf --min-dpi 300 --strict
```

### 中文配置

```python
setup_style(lang='zh')
# 衬线字体
setup_style(lang='zh', serif_for_zh=True)
```

## 实践建议

1. **科研作图必装**：如果你经常需要画论文图表，这个技能的「先分析再画图」理念比直接写 matplotlib 代码高效得多
2. **防错机制很实用**：自动拒绝不恰当的图表类型（饼图/3D/小样本柱状图），能帮你避免审稿人常见的可视化批评
3. **中文支持完善**：CJK 字体自动配置和回退链，中文论文用户友好
4. **关注 SciPilot 家族**：cite/writing 已发布，review/submit/read 计划中，值得持续关注整套组合
5. **与 SKB-002 互补**：Academic Research Skills 负责研究和写作流程，本技能专注于数据可视化环节，可以配合使用

## 安装状态

- **兼容性**：Claude Code 原生兼容 ✅
- **Python 要求**：3.9+
- **当前环境安装**：受网络限制，需在本地执行安装命令

## 关联条目

- [SKB-001 Codex Visio Paper Figure Skill](001_codex_visio_paper_figure_skill.md) - 同为图表/可视化工具，但面向 Visio
- [SKB-002 Academic Research Skills](002_academic_research_skills.md) - 学术研究全流程，可与本技能互补
