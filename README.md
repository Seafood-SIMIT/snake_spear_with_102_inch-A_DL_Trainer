# Snake Spear with 102 inch
A DeepLearning Trainer
一个简单的一维信号分类器的训练程序，一开始继承自谷歌的voicefilter，已经改的面目全非了。但是基本功能还是能用的。
## Usage
### 验证模式
用于验证模型的有效性，只读取两个文件进行快速训练并不保存checkpoint。
如验证数据量较少，自己设定batch_size.
'''
python trainer.py -c config/balabala.yaml -m name -v True
'''


### 正式模式
'''
python trainer.py -c config/balabala.yaml -m name 
'''