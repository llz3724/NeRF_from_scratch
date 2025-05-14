# train
```
python3 train.py --config=hparams.json
```

先预处理出所有图片的所有光线

设要迭代 200000 次，迭代一次是从所有光线里面有放回抽一个 batch ，采样后询问 MLP 部分，再渲染得到预期的像素，根据和实际的差别来调整 MLP 参数

实际上跑到 100000 次就差不多收敛了

kaggle 上面用 P100 训了大约 7h

all_pic 存了最终渲染出来的结果

kaggle_model 是kaggle训练100000次保存的模型

hparams.json 存超参数

load_blender.py 加载blender数据，暂时没有用真实数据

sample_utils.py 实现层级采样

render_utils.py 实现体渲染

model.py 实现位置编码和MLP

train.py 用于训练

view.py 渲染最终所有图片

kaggle 部署要仔细调整文件读写和目录等等，很麻烦，要小心，而且 kaggle 不能在线修改，写错了得重新上传