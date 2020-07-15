for k in model.state_dict():
    print(k)
    
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())

