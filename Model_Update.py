# 自己定义新的网络 然后把训练好的模型参数拿来使用

pre_model = pre_model() #之前的模型结构的初始化
 
pre_model.load_state_dict(torch.load('__.pth')) #之前的模型的加载
 
model=model() #现在模型结构
 
pre_dict = {k: v for k, v in pre_model.items() if k in model.state_dict()} #把resnet20的参数放进去，其他的参数还是对应的随机初始化的参数
 
model.load_state_dict(model.state_dict().update(pre_dict))#加载模型（模型的随机初始化模型（部分更新为预训练的模型））
