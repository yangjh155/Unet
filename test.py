import torch 

if __name__=="__main__":
    a = torch.randn(1, 1, 3, 4)
    print(a.shape)
    a = a.squeeze(0)
    print(a.shape)