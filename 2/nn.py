from numpy import NaN
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
SGD + BGD
'''

class MySigmoid(nn.Module):
    def __init__(self):
        super(MySigmoid, self).__init__()
        self.output = None

    def forward(self, input):
        output = 1 / (1 + torch.exp(-input))
        self.output = output
        return output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class MyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinear,self).__init__()
        self.W = nn.Parameter(torch.randn((input_dim, output_dim)), requires_grad=False)
        self.b = nn.Parameter(torch.randn(output_dim), requires_grad=False)
        self.gradW = 0
        self.gradb = 0
        self.input = None # (batch_size, input_dim)

    def forward(self, input):
        self.input = input
        output = input @ self.W + self.b
        return output

    def backward(self, grad_output):
        self.gradW = self.input.T @ grad_output
        self.gradb = grad_output.sum(dim=0)
        grad_input = grad_output @ self.W.T
        return grad_input
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.gradW
        self.b -= learning_rate * self.gradb

class MyModel(nn.Module):
    def __init__(self, num_feature, num_class, hidden_dim):
        super(MyModel, self).__init__()
        self.linear1 = MyLinear(num_feature, hidden_dim)
        self.sigmoid1 = MySigmoid()
        self.linear2 = MyLinear(hidden_dim, num_class)
        self.sigmoid2 = MySigmoid()


    def forward(self, input):
        z1 = self.linear1.forward(input)
        a1 = self.sigmoid1.forward(z1)
        z2 = self.linear2.forward(a1)
        a2 = self.sigmoid2.forward(z2)
        return a2
    
    def backward(self, grad_output):
        grad_z2 = self.sigmoid2.backward(grad_output)
        grad_a1 = self.linear2.backward(grad_z2)
        grad_z1 = self.sigmoid1.backward(grad_a1)
        self.linear1.backward(grad_z1)

    def update(self, learning_rate):
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)

def myMSE(logits, label):
    label_cate = torch.zeros_like(logits).scatter_(1, label.view((-1, 1)), 1)
    N = len(label)
    loss = 0.5 * torch.sum((label_cate - logits).pow(2)) / N
    grad = logits - label_cate

    return loss, grad

if __name__ == '__main__':
    # 参数
    epochs = 15
    learning_rate = 0.015
    device = torch.device('cuda')
    torch.manual_seed(410)
    torch.cuda.manual_seed_all(410)
    torch.backends.cudnn.deterministic = True

    # data
    train_data = datasets.MNIST('../data', train = True, download = True,
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(0, 1)
                                ]))
    data_size = len(train_data)

    # train
    model = MyModel(28*28, 12, 80)
    model.to(device)
    step = 1
    # SGD
    loss_history1 = []
    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size = 1, shuffle = True)
    for epoch in range(epochs):
        print(f'======= epoch {epoch+1} =======')
        for x, y in tqdm(train_loader):
            # 加载到GPU
            x = x.view((len(y), -1)).to(device)
            y = y.to(device)
            # 前向
            logits = model.forward(x)
            # 损失
            loss, grad = myMSE(logits, y)
            loss_history1.append(loss.cpu())
            # 后向
            model.backward(grad)
            # 更新参数
            model.update(learning_rate)
            step += 1

    # BGD
    loss_history2 = []
    train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size = data_size, shuffle = True)
    for epoch in range(epochs):
        print(f'======= epoch {epoch+1} =======')
        for x, y in tqdm(train_loader):
            # 加载到GPU
            x = x.view((len(y), -1)).to(device)
            y = y.to(device)
            # 前向
            logits = model.forward(x)
            # 损失
            loss, grad = myMSE(logits, y)
            loss_history2.append(loss.cpu())
            # 后向
            model.backward(grad)
            # 更新参数
            model.update(5e-5)
            step += 1


    plt.plot(torch.arange(1, len(loss_history1) + 1), loss_history1, label='SGD')
    plt.plot(torch.arange(1, len(loss_history2) + 1)*data_size, loss_history2, label='BGD')
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('loss.png')