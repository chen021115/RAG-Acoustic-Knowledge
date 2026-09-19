from modelscope import snapshot_download

model_dir = snapshot_download(
    "Xorbits/bge-large-zh-v1.5",
    cache_dir="./models"
)

print("模型下载完成：")
print(model_dir)