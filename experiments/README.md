# Experiments

每个 experiment 都应该是一个自包含的 slice。

推荐目录结构：

```text
experiments/YYYY-MM-DD-short-name/
  README.md
  commands.sh
  env.md
  results.csv
  figures/
  raw/      # 默认被 git 忽略，除非手动 force add
```

规则：

- 一个 slice 只回答一个问题。
- 保存 exact commands 和 environment。
- raw logs 如果很大，就保存在本地或外部存储。
- 处理后的 tables、figures 和 written interpretation 可以提交到 Git。
- 每次只改变一个主要变量。
