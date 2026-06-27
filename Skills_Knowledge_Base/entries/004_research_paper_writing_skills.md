# Research Paper Writing Skills - ML/CV/NLP 论文写作技能包

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-004                                                           |
| 来源     | https://github.com/Master-cai/Research-Paper-Writing-Skills        |
| 收录日期 | 2026-06-27                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #claude-code #codex #gemini #论文写作 #ML #CV #NLP #学术    |
| 重要程度 | ★★★★☆                                                             |
| 版本     | 无版本号（6 commits）                                               |
| 许可证   | MIT                                                                |
| 热度     | 4.4k stars / 225 forks                                             |
| 兼容性   | Claude Code / Codex / Gemini - 多平台可安装 ✅                      |

## 核心摘要

> 源于彭思达教授（Prof. Peng Sida）开源学习笔记的论文写作技能包，专注 ML/CV/NLP 领域。提供分章节的写作指南（摘要、引言、相关工作、方法、实验、结论），强调"先理清叙事再修改句子"的工作流，包含段落清晰度检查、五维自审、声明-证据对齐验证等质量检查机制。支持 Claude Code、Codex、Gemini 三平台安装。

## 关键知识点

- **来源权威**：基于彭思达教授的论文写作笔记，经整理和结构化改编
- **6 步核心工作流**：理清叙事 → 分章节指导 → 逐段改写 → 反向大纲验证 → 声明-实验对齐 → 对抗式自审
- **9 条全局原则**：核心是"一段一消息"，要求明确的首句表态和因果/对比/结论式的句间衔接
- **多平台兼容**：同时支持 Claude Code（全局/项目级）、Codex、Gemini 三个平台
- **分章节参考文档**：覆盖论文全部主要章节（Abstract、Introduction、Related Work、Method、Experiments、Conclusion）+ Paper Review
- **轻量级**：仅含 SKILL.md + 8 个参考文档 + 1 个示例目录，无代码依赖

## 详细笔记

### 6 步核心工作流

| 步骤 | 说明 |
|------|------|
| 1. 理清叙事 | 编辑句子前先明确论文整体叙事线 |
| 2. 分章节指导 | 应用章节特定的写作指南 |
| 3. 逐段改写 | 每条消息只改写一段（one paragraph per message） |
| 4. 反向大纲 | 每完成一个章节后执行反向大纲验证 |
| 5. 声明-实验对齐 | 验证所有声明都有实验结果支撑 |
| 6. 对抗式自审 | 从审稿人视角进行投稿前自审 |

### 9 条全局原则

- 一段一消息（one paragraph for one message only）
- 首句明确表态（explicit first-sentence messaging）
- 句间衔接使用因果/对比/结论连接（cause/contrast/consequence）
- 段落清晰度三步验证
- 五维自审（贡献、清晰度、实验、评估、方法可靠性）
- 声明-证据对齐验证

### 输出要求

修订章节时需交付：
1. 章节大纲
2. 带角色标注的修订段落
3. 自审检查清单
4. 声明-证据映射表

### 参考文档（references/）

| 文件 | 内容 |
|------|------|
| abstract.md | 摘要写作指南 |
| introduction.md | 引言写作指导 |
| related-work.md | 相关工作写作参考 |
| method.md | 方法部分撰写参考 |
| experiments.md | 实验部分写作建议 |
| conclusion.md | 结论写作指南 |
| paper-review.md | 论文评审相关 |
| does-my-writing-flow-source.md | 文章流畅性资源 |
| examples/ | 示例子目录 |

### 仓库结构

```
research-paper-writing/
├── SKILL.md                # 核心工作流和使用规则
├── references/             # 分章节写作指南
│   ├── abstract.md
│   ├── introduction.md
│   ├── related-work.md
│   ├── method.md
│   ├── experiments.md
│   ├── conclusion.md
│   ├── paper-review.md
│   ├── does-my-writing-flow-source.md
│   └── examples/
└── agents/
    └── openai.yaml         # 代理元数据
```

## 代码/配置片段

### Claude Code 安装（全局）

```bash
mkdir -p "$HOME/.claude/skills"
cp -R research-paper-writing "$HOME/.claude/skills/"
```

### Claude Code 安装（项目级）

```bash
mkdir -p .claude/skills
cp -R research-paper-writing .claude/skills/
```

### Codex 安装

```bash
mkdir -p "$CODEX_HOME/skills"
cp -R research-paper-writing "$CODEX_HOME/skills/"
```

### Gemini 安装

```bash
mkdir -p "$HOME/.gemini/skills"
cp -R research-paper-writing "$HOME/.gemini/skills/"
```

### 使用方式

```
Use $research-paper-writing to improve my paper's Introduction.
```

### 典型用例

- 草稿或重写摘要/引言/方法/实验/结论
- 改进段落流畅性和章节逻辑
- 检查声明与证据的对应关系
- 从审稿人角度进行投稿前自审

## 实践建议

1. **ML/CV/NLP 论文写作首选**：如果你做这些领域的研究，这个技能包的写作方法论非常扎实（源自彭思达教授的笔记）
2. **轻量无依赖**：纯 Markdown 参考文档，无需安装任何额外依赖，复制即用
3. **多平台通用**：同一份技能包可以在 Claude Code、Codex、Gemini 三个平台使用
4. **与 SKB-002 定位不同**：SKB-002（Academic Research Skills）是全流程自动化套件（研究→写作→评审→修订），本技能更侧重写作方法论和技巧指导，两者可以互补
5. **与 SKB-003 配合**：SciPilot Figure Skill 负责图表，本技能负责文字，覆盖论文制作的不同环节
6. **逐段改写策略好用**：一段一消息的工作方式可以让 AI 产出更精准，值得在日常写作中应用

## 安装状态

- **兼容性**：Claude Code 原生兼容 ✅（同时兼容 Codex、Gemini）
- **依赖**：无额外依赖，纯 Markdown
- **当前环境安装**：受网络限制，需在本地执行 git clone 后复制

## 关联条目

- [SKB-002 Academic Research Skills](002_academic_research_skills.md) - 全流程自动化套件，本技能侧重写作方法论
- [SKB-003 SciPilot Figure Skill](003_scipilot_figure_skill.md) - 图表可视化，与本技能互补
