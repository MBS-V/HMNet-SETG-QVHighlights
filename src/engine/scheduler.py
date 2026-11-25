import torch

def build_scheduler(optimizer,warmup_epochs,total_epochs):
    def scale(epoch):
        if epoch<warmup_epochs: return float(epoch+1)/max(1,warmup_epochs)
        progress=(epoch-warmup_epochs)/max(1,total_epochs-warmup_epochs)
        return 0.5*(1+__import__('math').cos(__import__('math').pi*min(1.0,progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
