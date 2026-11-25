# pairwise-clearcoat-flake-glint

## 领域

游戏、图形与多媒体

## 主题

车漆清漆层与金属片闪烁

## 说明

掠射菲涅尔、清漆法线独立、切线空间 flake。

初始状态下 `graphics/clearcoat_flake.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
