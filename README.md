# Manga OCR Server (for SuwayomiGO)


## 使用说明：
- 功能：接收客户端请求，OCR识别日漫文本，并返回中文翻译和词典数据（支持AI翻译）
- 本客户端需搭配漫画阅读器 [SuwayomiGO](https://github.com/lightningandme/SuwayomiGO) 和漫画服务器 [Suwayomi-Server
](https://github.com/Suwayomi/Suwayomi-Server) 使用
- 拷贝仓库代码，使用python3.10，根据requirements.txt安装依赖
- 打开 suwayomigo_service 根目录
- 运行 server.py 即可启动服务器，首次启动会下载一些模型，请耐心等待
- 如需配置AI翻译，请根据.env.example指引
- 如需更换本地词典，请将Yomitan词典的zip放到 for_dict 目录，运行 convert_yomitan.py，用新生成的 manga_dict.db 替换掉根目录同名文件
- 等版本功能稳定后，将考虑发放一键安装包😊

### Acknowledgments

This project was done with the usage of:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [CC-100](https://data.statmt.org/cc-100/) dataset
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [manga-ocr](https://github.com/kha-white/manga-ocr)
