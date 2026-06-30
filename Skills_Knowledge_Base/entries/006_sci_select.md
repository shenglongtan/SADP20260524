# sci-select - SCI 期刊选择 AI 技能

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-006                                                           |
| 来源     | https://github.com/keros68/sci-select                              |
| 收录日期 | 2026-06-30                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #claude-code #期刊选择 #SCI #投稿 #Python #学术            |
| 重要程度 | ★★★★☆                                                             |
| 版本     | 无版本号                                                           |
| 许可证   | MIT                                                                |
| 热度     | 15 stars / 4 forks                                                 |
| 兼容性   | Claude Code / Codex - 可安装 ✅                                     |

## 核心摘要

> 面向科研人员的 SCI/SCIE/ESCI/SSCI 期刊选择 AI 技能。输入论文标题、摘要、关键词或正文片段，自动识别研究主题，检索 LetPub 候选期刊，聚合 IF、CAS 分区、XinRui 分区、审稿周期、OA/APC 等多维指标，输出四级投稿推荐带（ambitious/solid/safer/cautious）。支持本地期刊索引、OpenAlex 和可选 XinRui API。

## 关键知识点

- **四级推荐带**：ambitious（冲刺）/ solid（稳妥）/ safer（保险）/ cautious（谨慎）
- **多源指标聚合**：本地索引 + LetPub + OpenAlex + 可选 XinRui API
- **输出指标覆盖**：IF、2025 CAS 分区、2026 XinRui 分区、SCI 类型、审稿周期、h-index、OA/APC 状态
- **风险标记**：ESCI 状态、重审稿周期期刊、弱匹配、数据缺口
- **官方 Journal Finder 链接**：支持 Elsevier、Springer Nature、Wiley、Taylor & Francis 四大出版商
- **明确不做什么**：不预测录用率、不替代官网、不自动登录、不抓取论坛数据

## 详细笔记

### 核心工作流

```
论文元数据 → 主题/方法/证据提取 → LetPub 候选检索
→ 指标聚合（本地+LetPub+OpenAlex±XinRui）
→ 按研究范围/证据/风险排序 → 四级推荐输出
```

### 四级推荐带

| 推荐带 | 说明 |
|--------|------|
| ambitious | 冲刺目标，影响力高，录取竞争激烈 |
| solid | 稳妥选择，主题匹配度高 |
| safer | 较保险，接受率相对较高 |
| cautious | 保守选择，审稿周期/风险需关注 |

### 输出报告包含内容

- 已知期刊指标摘要
- 识别出的研究方向和主题匹配
- 决策矩阵
- 推荐层级和投稿带
- 各项指标：IF、2025 CAS、2026 XinRui、SCI 类型、审稿速度、h-index、OA/APC
- 主题匹配理由说明
- 风险标记：ESCI、高拒稿周期、弱匹配、数据缺口
- 数据来源说明

### 仓库结构

```
sci-select/
├── SKILL.md                          # 技能主文档
├── agents/openai.yaml                # UI 元数据
├── requirements.txt                  # Python 依赖
├── scripts/
│   ├── select_journals.py            # 主题识别、检索、排序、报告
│   ├── journal_metrics.py            # 期刊查询和指标聚合
│   ├── journal_index_client.py       # 本地/静态索引读取
│   ├── official_finders.py           # 官方 Journal Finder 工具
│   ├── letpub_client.py              # LetPub 公开页面抓取
│   └── recommend.py                  # 旧版接口封装
├── references/
│   └── data-sources.md               # 数据来源文档
├── examples/
│   └── demo-report.md                # 示例选刊报告
└── tests/
```

## 代码/配置片段

### 安装

```bash
# Claude Code
git clone https://github.com/keros68/sci-select.git \
  ~/.claude/skills/sci-select

# Codex
git clone https://github.com/keros68/sci-select.git \
  ~/.codex/skills/sci-select

# 安装依赖
pip install -r requirements.txt
```

### 环境变量配置

```bash
# 可选：本地期刊索引
export SCI_SELECT_JOURNAL_INDEX_PATH="/path/to/search_index.json"
# 或
export SCI_SELECT_JOURNAL_INDEX_URL="https://example.com/search_index.json"

# 可选：XinRui API（新睿分区）
export XINRUI_API_KEY="YOUR_API_KEY"
```

### Python API 示例

```python
# 查询单个期刊指标
from scripts.journal_metrics import get_journal_metrics, format_metrics_line
metrics = get_journal_metrics("Journal of Hydrology")
print(format_metrics_line(metrics))

# 运行选刊工作流
from scripts.select_journals import select_journals, format_selection_report
paper_text = """论文标题 + 摘要 + 关键词"""
bundle = select_journals(text=paper_text, impact_low="3", max_candidates=8)
print(format_selection_report(bundle["profile"], bundle["results"]))

# 生成官方 Journal Finder 链接
from scripts.official_finders import build_finder_checklist, format_finder_checklist
checklist = build_finder_checklist(title="论文标题", abstract=paper_text, keywords=["关键词1"])
print(format_finder_checklist(checklist))
```

### 本地期刊索引格式

```json
{
  "meta": {"source": "local"},
  "journals": [
    {
      "title": "ENVIRONMENTAL POLLUTION",
      "issn": "0269-7491",
      "cas_2025": "2区",
      "xuankan_2026": "2区"
    }
  ]
}
```

## 实践建议

1. **投稿前必备工具**：写完论文不知道投哪里？用这个技能自动筛选匹配期刊并分级推荐
2. **热度不高但功能扎实**：15 stars 不代表质量差，功能覆盖（四级推荐+多源指标+风险标记）相当完整
3. **配置本地索引效果更好**：提供本地 CAS/XinRui 分区数据后，精准度大幅提升
4. **与 SKB-005 形成完整投稿前准备链**：PaperJury 做论文质量压测，sci-select 做目标期刊筛选，两者组合覆盖投稿准备全流程
5. **注意限制**：不要直接引用工具输出的"CAS 2026 分区"，需人工到官网核实最新分区

## 安装状态

- **兼容性**：Claude Code 原生兼容 ✅
- **依赖**：Python + requirements.txt，可选 XinRui API Key
- **当前环境安装**：受网络限制，需在本地执行安装命令

## 关联条目

- [SKB-005 PaperJury](005_paperjury.md) - 投稿前论文质量压测，与本技能组合覆盖投稿准备全流程
- [SKB-002 Academic Research Skills](002_academic_research_skills.md) - 全流程学术研究套件
