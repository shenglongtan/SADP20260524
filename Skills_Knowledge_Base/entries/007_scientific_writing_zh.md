# Scientific-Writing-zh - 中文科研写作教练技能

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-007                                                           |
| 来源     | https://github.com/YuanZHAO321/Scientific-Writing-zh              |
| 收录日期 | 2026-07-02                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #claude-code #科研写作 #中文 #论文写作 #ESL #学术写作教练   |
| 重要程度 | ★★★★☆                                                             |
| 版本     | 无版本号（3 commits）                                               |
| 许可证   | MIT                                                                |
| 热度     | 12 stars / 1 fork                                                  |
| 兼容性   | Claude Code - 可安装 ✅                                             |

## 核心摘要

> 面向中文研究者的英文科研写作教练 Claude Code 技能。设计理念是"用中文讲解原则，例句保留英文"（因中文研究者多投英文国际期刊）。覆盖科研写作全生命周期：从发表决策、初稿撰写，到文字修磨、审稿回复、会议演示、职场沟通。通过渐进披露设计，只在任务匹配时加载相关参考文件。

## 关键知识点

- **定位精准**：专为中文母语研究者写英文论文而设计，双语教学（中文解释+英文例句）
- **全生命周期覆盖**：12 个参考文档涵盖科研写作从头脑风暴到论文推广的完整流程
- **两大核心方法**：写作/编辑漏斗（文档→段落→句子→词语→标点）、CPR 修改法（Concision/Precision/Revision）
- **触发方式**：无需显式调用，用中文提问时自动触发
- **渐进披露设计**：仅在任务匹配时加载相关参考文件，保持轻量
- **轻量无依赖**：纯 Markdown，无代码依赖，git clone 即用

## 详细笔记

### 覆盖的科研写作场景

| 参考文件 | 覆盖内容 |
|----------|---------|
| pre-writing.md | 动机激发、头脑风暴、提纲制定、初稿撰写 |
| publishing-process.md | 发表决策、期刊选择、校样处理、论文推广 |
| paper-structure.md | 标题、摘要、引言、方法、结果、讨论、结论 |
| prose-craft.md | 段落、句子、词语层级的文字修磨 |
| revision.md | 修改策略（编辑漏斗、CPR 方法、凝练） |
| figures-citations.md | 图表、公式、引用管理 |
| authorship-ethics.md | 署名、作者顺序、学术伦理 |
| esl-guidance.md | 非英语母语作者专项指南 |
| peer-review.md | 同行评审意见撰写与回复 |
| presentations.md | 会议摘要、演示、报告、海报 |
| career-communication.md | 职场沟通、邮件、媒体交流 |
| word-usage.md | 词汇、标点、常见误用 |

### 两大核心方法

**写作/编辑漏斗**：从大尺度到小尺度递进修改
```
文档整体 → 段落结构 → 句子表达 → 词语选择 → 标点细节
```

**CPR 修改法**：
- **C**oncision（简洁）：去除冗余
- **P**recision（精确）：选词准确
- **R**evision（修订）：系统性改稿

### 仓库结构

```
scientific-writing-zh/
├── SKILL.md                 # 路由与核心方法（技能入口）
├── README.md
├── LICENSE (MIT)
└── references/
    ├── pre-writing.md
    ├── publishing-process.md
    ├── paper-structure.md
    ├── prose-craft.md
    ├── revision.md
    ├── figures-citations.md
    ├── authorship-ethics.md
    ├── esl-guidance.md
    ├── peer-review.md
    ├── presentations.md
    ├── career-communication.md
    └── word-usage.md
```

## 代码/配置片段

### 安装

```bash
git clone https://github.com/YuanZHAO321/scientific-writing-zh.git \
  ~/.claude/skills/scientific-writing-zh
```

### 验证安装

```bash
# 重启 Claude Code 后执行
/skills
```

### 触发示例

```
# 无需特殊命令，直接用中文提问：
帮我把这段改紧凑些，去掉被动语态

这个摘要够具体吗？是不是没超250词？

帮我回复审稿人2——他们觉得方法没讲清楚

给这篇稿子写同行评审意见

我的引言不连贯——先把结构理顺
```

## 实践建议

1. **中文研究者首选**：比 SKB-004（英文写作技能包）更适合母语是中文的研究者，解释语言是中文，学习曲线更低
2. **ESL 专项指南是亮点**：`esl-guidance.md` 专门针对非母语英文写作常见问题，很有针对性
3. **与 SKB-006 配合**：sci-select 选好期刊目标后，用本技能针对目标期刊的写作风格优化论文
4. **与 SKB-005 配合**：本技能优化写作质量，PaperJury 做最终压力测试
5. **轻量易上手**：纯 Markdown 无依赖，适合快速部署在所有 Claude Code 环境

## 安装状态

- **兼容性**：Claude Code 原生兼容 ✅
- **依赖**：无额外依赖，纯 Markdown
- **当前环境安装**：受网络限制，需在本地执行安装命令

## 关联条目

- [SKB-004 Research Paper Writing Skills](004_research_paper_writing_skills.md) - 英文版写作技能包（含 ML/CV/NLP 专项），与本技能定位互补
- [SKB-005 PaperJury](005_paperjury.md) - 投稿前压力测试，可作为本技能的下游
- [SKB-006 sci-select](006_sci_select.md) - 期刊选择，与本技能配合使用
