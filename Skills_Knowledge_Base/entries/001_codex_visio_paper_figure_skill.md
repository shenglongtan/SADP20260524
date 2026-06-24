# Codex Visio Paper Figure Skill - 学术论文图表 Visio 重建技能

| 字段     | 值                                                                 |
|----------|-------------------------------------------------------------------|
| ID       | SKB-001                                                           |
| 来源     | https://github.com/pengjunchi0/codex-visio-paper-figure-skill     |
| 收录日期 | 2026-06-23                                                        |
| 分类     | SK - Skills/技能                                                   |
| 标签     | #skills #codex #visio #论文图表 #PowerShell #COM自动化 #图表重建    |
| 重要程度 | ★★★★☆                                                             |
| 版本     | v1.1.1                                                            |
| 语言     | PowerShell (100%)                                                  |
| 热度     | 251 stars / 13 forks                                               |

## 核心摘要

> 这是一个 Codex Skill，用于将参考图片（PNG/JPG/截图）或 AI 生成的论文模型图重建为可编辑的 Microsoft Visio `.vsdx` 原生格式。核心理念是使用 Visio 原生形状（矩形、圆、线条、箭头、连接线、文本）来重建图表，严禁将整张参考图片嵌入作为最终内容。支持导出为 SVG、PDF、PPTX、PNG 多种格式。

## 关键知识点

- **Skill 安装方式**：克隆仓库到 Codex skills 目录（`$env:USERPROFILE\.codex\skills\visio-image-rebuilder`），重启 Codex 即可自动发现
- **核心文件是 `SKILL.md`**：这是 Skill 的主入口，包含触发描述、工作流、验收标准、安全规则
- **依赖 Windows + Visio COM Automation**：完整自动绘图功能需要 Windows 系统和 Microsoft Visio
- **面板校准机制（v1.1.1新增）**：对复杂多面板图表，使用局部坐标系（0-1）而非全局坐标，防止元素溢出和重叠
- **辅助函数**：`RectRel`、`TextRel`、`OvalRel`、`LineRel` 用于局部坐标映射；`Assert-RelBox`、`Assert-RelPoint` 用于边界溢出检测
- **严格验收标准**：布局一致、模块完整、文本可编辑、无整图嵌入、样式统一、无跨面板重叠

## 详细笔记

### 仓库结构

```
.
├── README.md
├── SKILL.md                              # Skill 主入口
├── agents/
│   └── openai.yaml                       # Codex UI 元数据
├── references/
│   └── rebuild-guidelines.md             # 复杂科学图表重建指南
└── scripts/
    ├── visio_export_formats.ps1          # 导出函数（PNG/SVG/PDF/PPTX）
    ├── visio_page_tools.ps1              # 辅助检查脚本（备份、导出、包结构检查）
    └── visio_rebuild_scaffold.ps1        # Visio 原生绘图脚手架
```

### 适用场景

| 适用 | 不适用 |
|------|--------|
| 从 PNG/JPG/截图重建 Visio 图表 | 简单图片插入 Visio |
| AI 生成的论文模型图转可编辑 .vsdx | 通用图片编辑/增强 |
| 修改现有 Visio 文件的配色/布局 | 不需要原生可编辑性的光栅复制 |
| 复杂多面板科学图表的结构复制 | |
| 标准化论文图表样式 | |
| 导出为 SVG/PDF/PPTX/PNG | |

### 多面板校准工作流（v1.1.1 重点）

1. 校准参考图片与 Visio 页面整体尺寸
2. 定义每个主面板的左上角、宽度、高度，必要时记录角点
3. 使用 0-1 局部坐标绘制面板内部元素（非全局坐标）
4. 导出预览并验证：无元素溢出、无相邻面板重叠、无箭头/文本越界

### 版本历史

| 版本 | 重点 |
|------|------|
| v1.0 | 基础 Skill 框架、原生形状可编辑性、Visio COM 绘图脚手架 |
| v1.1 | 多格式导出支持（PNG/SVG/PDF/PPTX） |
| v1.1.1 | 面板校准和防重叠机制 |
| v1.2（计划中） | 更多图形辅助函数（DrawCube、DrawHeatmap、DrawGraph 等） |

## 代码/配置片段

### 安装命令

```powershell
git clone https://github.com/pengjunchi0/codex-visio-paper-figure-skill.git "$env:USERPROFILE\.codex\skills\visio-image-rebuilder"
```

### 导出现有 .vsdx

```powershell
powershell -ExecutionPolicy Bypass -File scripts\visio_page_tools.ps1 `
  -VsdxPath "C:\path\model.vsdx" `
  -ExportFormats svg,pdf,pptx `
  -OutputDir "C:\path\exports" `
  -InspectPackage
```

### 重建并导出

```powershell
powershell -ExecutionPolicy Bypass -File scripts\visio_rebuild_scaffold.ps1 `
  -VsdxPath "C:\path\model.vsdx" `
  -PageW 16 `
  -PageH 9 `
  -RefW 1600 `
  -RefH 900 `
  -PreviewPath "C:\path\exports\model.png" `
  -ExportFormats svg,pdf,pptx `
  -OutputDir "C:\path\exports"
```

### 使用示例（自然语言触发）

```
使用 visio-image-rebuilder，根据这张参考图片重建 C:\path\model.vsdx，
要求最终是 Visio 原生可编辑形状，不要整图嵌入，并导出 SVG、PDF、PPTX。
```

```
把这个 .vsdx 按参考图更换配色，保持布局不变，最终仍然可编辑，
并给我一个 PDF 预览和 PPTX。
```

## 实践建议

1. **环境要求高**：该技能强依赖 Windows + Microsoft Visio，Linux/macOS 用户无法使用完整功能
2. **学术论文场景价值大**：如果经常需要制作/修改论文中的模型框架图、流程图，这个技能非常实用
3. **Skill 架构值得参考**：即使不用 Visio，该仓库的 Skill 组织方式（SKILL.md 入口 + agents yaml + references + scripts）是一个很好的 Codex Skill 模板
4. **面板校准思路可复用**：局部坐标系 + 边界检测的思路可以应用到其他图表生成场景
5. **注意版本发展**：v1.2 计划添加更多图形辅助函数，值得持续关注

## 关联条目

- [SKB-002 Academic Research Skills](002_academic_research_skills.md) - 同为学术论文工具链
