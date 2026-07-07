# ASTRA Plotter 文档

本目录包含项目文档。

## 编译 LaTeX 手册

```bash
cd docs

# 使用 xelatex 编译（支持中文）
xelatex manual.tex
xelatex manual.tex  # 运行两次生成目录

# 或使用 latexmk
latexmk -xelatex manual.tex
```

编译输出：`manual.pdf`
