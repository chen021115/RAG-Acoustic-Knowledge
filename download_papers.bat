@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo 水声 RAG 文献一键下载（默认前20篇）
echo ============================================
python download_papers.py --limit 20

echo.
echo 下载结束。PDF 将保存在 docs 下的分类目录中。
echo 如需全部25篇，请运行：python download_papers.py --limit 25
echo.
pause
