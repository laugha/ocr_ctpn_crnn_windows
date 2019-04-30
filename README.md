# ocr_ctpn_crnn_windows
chinese+english+numbers ocr algorithm based on ctpn+crnn especially on Windows
# 前言
我在github上找了半天关于中文ocr的代码，大部分都是linux环境下的，windows下运行的几乎没有。
费了好大功夫才将文本检测ctpn算法和文字识别crnn算法在windows运行，并将两者集合起来，形成一个
对一张照片的ocr识别。
# 运行环境
请阅读requirement.txt，其中tf和torch都需要，tf的版本要1.7.0不能安装最新的
# 测试
ctpn部分测试 --> crnn_test.py
crnn部分测试 --> ctpn_test.py
整体测试 test.py
# 预训练模型
链接：https://pan.baidu.com/s/1Yjf3qMi82Qmy2vGTTuHZmA 
提取码：i753 
# 最终
如果有问题或者对ocr算法有改进建议，请联系我的邮箱84019582@qq.com
