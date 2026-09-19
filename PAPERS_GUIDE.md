# 水声 RAG 文献语料包

本项目附带 `papers_manifest.csv`，整理了 25 篇与水声定位、多径传播、匹配场处理、DOA、OFDM 信道估计和 SSP 等相关的公开论文。

## 一键下载

在项目根目录执行：

```bash
python download_papers.py
```

默认尝试下载前 20 篇。若希望下载全部 25 篇：

```bash
python download_papers.py --limit 25
```

下载后的目录结构：

```text
docs/
├── 01_localization/
├── 02_multipath_mfp/
├── 03_signal_processing/
├── 04_channel_communication/
└── 05_surveys/
```

如果出版社临时启用反爬、验证码或维护页，脚本会将该条标为 `FAIL`，并打印论文的公开落地页。此时浏览器打开该页面点击 `Download PDF` 即可，不会把 HTML 错误页伪装成 PDF 保存下来。

## 建议第一批用于 RAG 的论文

不建议第一次就塞满 25 篇。先下载 15–20 篇，重点包含：

- TDOA/FDOA 水声定位；
- 多径条件下的 TDOA 与 NLOS 定位；
- Matched Field Processing（MFP）；
- Time Reversal / BELLHOP；
- DOA / Capon / 阵列信号处理；
- UWA-OFDM 与信道估计；
- SSP 测量与反演综述。

这样能让知识库既覆盖你当前的水声定位研究，也能测试跨论文检索和引用能力。
