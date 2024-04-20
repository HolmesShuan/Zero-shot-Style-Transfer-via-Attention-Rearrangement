## 项目名称
Z-STAR: 通过注意力重排实现零样本风格迁移

## 简介
Z-STAR 是一种新颖的零样本（训练自由）风格迁移方法，它利用预训练的扩散模型中的先验知识，通过注意力重排策略，实现内容和风格的有效融合。该方法不需要对每个输入风格进行重新训练或调整，能够直接从目标风格图像中提取风格信息，并将其无缝融合到内容图像中。

## 主要贡献
- 提出了一种不需要重新训练或调整的零样本图像风格迁移方法。
- 引入了一种重排注意力机制，用于在扩散潜在空间中解耦和融合内容/风格信息。

## 安装
```bash
# comming soon...
```

## 使用方法
```bash
python main.py --content_image <path_to_content_image> --style_image <path_to_style_image>
```

## 论文和代码
本项目是论文 "Z∗: Zero-shot Style Transfer via Attention Rearrangement" 的官方实现，详情请查看 [论文链接](https://arxiv.org/abs/2311.16491)。

## 许可证
本项目遵循 **Apache license 2.0** 许可证。可以商用。
