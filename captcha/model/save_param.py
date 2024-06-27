import torch
def save_param(epoch, model_name ,state_dict, dir):
    import time
    tm = time.localtime(time.time())
    file_name = f'/{model_name}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{epoch}.pth'
    torch.save(state_dict, dir + file_name)