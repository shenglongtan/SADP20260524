# Academic Research Skills - 学术研究全流程技能套件

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-002                                                           |
| 来源     | https://github.com/Imbad0202/academic-research-skills              |
| 收录日期 | 2026-06-24                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #claude-code #学术研究 #论文写作 #文献综述 #同行评审 #pipeline |
| 重要程度 | ★★★★★                                                             |
| 版本     | v3.13.0                                                           |
| 许可证   | CC-BY-NC 4.0                                                      |
| 热度     | 34.1k stars / 2.8k forks                                          |
| 兼容性   | Claude Code (v3.7.0+) - 可安装                                     |

## 核心摘要

> 一套覆盖学术研究全流程的 Claude Code 技能套件，包含 4 个核心技能：Deep Research（深度研究）、Academic Paper（论文写作）、Academic Paper Reviewer（同行评审）、Academic Pipeline（全流程编排）。强调人类监督而非全自动化，内置完整性验证门和多阶段审查流程。支持中英日韩多语言，支持 APA/Chicago/MLA/IEEE/Vancouver 多种引用格式。

## 关键知识点

- **4 个核心技能**：Deep Research / Academic Paper / Academic Paper Reviewer / Academic Pipeline
- **10 阶段流水线**：研究 → 写作 → 完整性检查 → 评审 → 修订 → 格式化 → 总结，含 2 个强制完整性门（Stage 2.5 和 4.5）
- **安装方式**：支持 plugin marketplace 一键安装（`/plugin marketplace add`）和传统 git clone 安装
- **Multi-Agent 架构**：Deep Research 使用 13-agent 团队，Academic Paper 使用 12-agent 写作流水线，Reviewer 使用 7-agent 多视角评审
- **引用验证三角化**：通过 Semantic Scholar + OpenAlex + Crossref + arXiv 四源交叉验证引文真实性
- **Material Passport**：跨阶段的元数据追踪中心，包含文献语料、实验溯源、合规历史等
- **反谄媚机制**：Devil's Advocate 使用让步阈值协议，不会在压力下软化立场
- **预算估算**：每篇 15k 字论文约 $4-6 USD

## 详细笔记

### 四大核心技能

#### 1. Deep Research (v2.11.0)
| 模式 | 说明 |
|------|------|
| full | 完整研究 |
| quick | 快速调研 |
| review | 文献审查 |
| lit-review | 文献综述 |
| three-way-scan | 三方对比扫描 |
| fact-check | 事实核查 |
| socratic | 苏格拉底引导模式 |
| systematic-review | PRISMA 系统综述 |

特点：13-agent 研究团队、意图检测、对话健康度监控、Semantic Scholar API 验证

#### 2. Academic Paper (v3.2.0)
| 模式 | 说明 |
|------|------|
| full | 完整写作 |
| plan | 计划 |
| outline-only | 仅大纲 |
| revision | 修订 |
| revision-coach | 修订教练 |
| abstract-only | 仅摘要 |
| lit-review | 文献综述 |
| format-convert | 格式转换 |
| citation-check | 引文检查 |
| disclosure | 披露声明 |
| rebuttal-audit | 反驳审计 |

特点：12-agent 写作流水线、风格校准、LaTeX 支持（APA 7.0）、输出 Markdown/DOCX/PDF

#### 3. Academic Paper Reviewer (v1.10.0)
| 模式 | 说明 |
|------|------|
| full | 完整评审 |
| re-review | 重新评审 |
| quick | 快速评审 |
| methodology-focus | 方法论聚焦 |
| guided | 引导式 |
| calibration | 校准 |

评分标准：≥80 Accept / 65-79 Minor Revision / 50-64 Major Revision / <50 Reject

#### 4. Academic Pipeline (v3.13.0) - 10 阶段编排器

| 阶段 | 功能 |
|------|------|
| 1 | RESEARCH - 文献调研 |
| 2 | WRITE - 手稿撰写 |
| 2.5 | INTEGRITY GATE - 声明验证（强制） |
| 3 | REVIEW - 同行评审第一轮 |
| 3' | RE-REVIEW - 修订验证 |
| 4 | REVISE - 作者回应与修订 |
| 4.5 | INTEGRITY GATE - 最终验证（强制） |
| 5 | FORMAT - 出版准备 |
| 6 | SUMMARY - 协作评估 |

### Slash Commands 速查

```
/ars-plan              # 苏格拉底式论文规划引导
/ars-lit-review        # 快速文献调研
/ars-research          # 完整研究阶段
/ars-write             # 完整论文写作
/ars-review            # 同行评审会话
/ars-full-pipeline     # 端到端全流程
/ars-3w               # 三方论文对比
/ars-rebuttal-audit   # 反驳稿验证
/ars-cache-invalidate # 清除验证缓存
/ars-mark-read        # 确认咨询发现
```

### 支持的引用格式

| 格式 | 说明 |
|------|------|
| APA 7.0 | 默认，含中文规则 |
| Chicago | Notes & Author-Date |
| MLA | |
| IEEE | |
| Vancouver | |

### 支持语言

英语、繁体中文、简体中文、日语、韩语

### 仓库结构

```
academic-research-skills/
├── .claude-plugin/          # Plugin manifest
├── .claude/                 # 配置文件
├── skills/                  # 四个技能目录
│   ├── deep-research/
│   ├── academic-paper/
│   ├── academic-paper-reviewer/
│   └── academic-pipeline/
├── commands/                # Slash command 定义
├── agents/                  # Plugin 附带的 agents
├── hooks/                   # PreToolUse/PostToolUse 处理器
├── docs/
│   ├── ARCHITECTURE.md      # 流水线结构参考
│   ├── SETUP.md             # 安装与前置条件
│   └── PERFORMANCE.md       # Token 预算与成本
├── scripts/                 # 工具脚本与验证器
├── examples/showcase/       # 真实流水线产物
└── shared/
    ├── handoff_schemas.md   # Material Passport 规范
    └── references/          # 操作协议
```

## 代码/配置片段

### 一键安装（推荐）

```bash
/plugin marketplace add Imbad0202/academic-research-skills
/plugin install academic-research-skills
```

### 传统安装

```bash
git clone https://github.com/Imbad0202/academic-research-skills
# 然后 symlink 到 ~/.claude/skills/
```

### 前置条件

**必需：**
- Claude Code CLI (v3.7.0+) 或 VS Code/JetBrains 扩展
- `ANTHROPIC_API_KEY` 环境变量

**可选：**
- Pandoc（DOCX/PDF 输出）
- tectonic（LaTeX PDF 编译）
- Source Han Serif TC 字体（中文 PDF 输出）
- Python 3.x（高级守卫和可选功能）
- Git Bash on Windows（hook 可移植性）

## 实践建议

1. **学术研究必装**：如果你经常做学术研究和论文写作，这是目前最完整的 Claude Code 技能套件（34.1k stars）
2. **从 /ars-plan 开始**：不要直接跳到写作，先用苏格拉底引导模式规划论文结构
3. **善用完整性门**：Stage 2.5 和 4.5 的强制验证是核心设计，不要试图跳过
4. **引文验证很关键**：四源交叉验证可以有效防止 AI 幻觉引文
5. **预算可控**：每篇约 $4-6，性价比很高
6. **Skill 架构值得深入学习**：`.claude-plugin/` + `skills/` + `commands/` + `hooks/` + `agents/` 的组织方式是 Claude Code Plugin 的标准范式

## 安装状态

- **兼容性**：Claude Code 原生兼容 ✅
- **当前环境安装**：需要用户在本地 Claude Code 中执行 `/plugin marketplace add` 命令

## 关联条目

- [SKB-001 Codex Visio Paper Figure Skill](001_codex_visio_paper_figure_skill.md) - 同为学术论文工具链
