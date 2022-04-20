from cProfile import label
from numpy import NaN
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
一层模型以及三层模型
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

class MyModel1Layer(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MyModel1Layer, self).__init__()
        self.linear1 = MyLinear(num_feature, num_class)
        self.sigmoid1 = MySigmoid()

    def forward(self, input):
        z1 = self.linear1.forward(input)
        a1 = self.sigmoid1.forward(z1)
        return a1
    
    def backward(self, grad_output):
        grad_z1 = self.sigmoid1.backward(grad_output)
        self.linear1.backward(grad_z1)

    def update(self, learning_rate):
        self.linear1.update(learning_rate)

class MyModel3Layer(nn.Module):
    def __init__(self, num_feature, num_class, hidden_dim):
        super(MyModel3Layer, self).__init__()
        self.linear1 = MyLinear(num_feature, hidden_dim)
        self.sigmoid1 = MySigmoid()
        self.linear2 = MyLinear(hidden_dim, hidden_dim)
        self.sigmoid2 = MySigmoid()
        self.linear3 = MyLinear(hidden_dim, num_class)
        self.sigmoid3 = MySigmoid()

    def forward(self, input):
        z1 = self.linear1.forward(input)
        a1 = self.sigmoid1.forward(z1)
        z2 = self.linear2.forward(a1)
        a2 = self.sigmoid2.forward(z2)
        z3 = self.linear3.forward(a2)
        a3 = self.sigmoid3.forward(z3)
        return a3
    
    def backward(self, grad_output):
        grad_z3 = self.sigmoid3.backward(grad_output)
        grad_a2 = self.linear3.backward(grad_z3)
        grad_z2 = self.sigmoid2.backward(grad_a2)
        grad_a1 = self.linear2.backward(grad_z2)
        grad_z1 = self.sigmoid1.backward(grad_a1)
        self.linear1.backward(grad_z1)

    def update(self, learning_rate):
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)
        self.linear3.update(learning_rate)

def myMSE(logits, label):
    label_cate = torch.zeros_like(logits).scatter_(1, label.view((-1, 1)), 1)
    N = len(label)
    loss = 0.5 * torch.sum((label_cate - logits).pow(2)) / N
    grad = logits - label_cate

    return loss, grad

if __name__ == '__main__':
    # 参数
    epochs = 15
    batch_size = 512
    learning_rate = 0.015
    device = torch.device('cuda')
    torch.manual_seed(410)
    torch.cuda.manual_seed_all(410)
    torch.backends.cudnn.deterministic = True

    # data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train = True, download = True,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0, 1)
                        ])),
                        batch_size = batch_size, shuffle = True)

    # train
    model1 = MyModel1Layer(28*28, 12)
    model1.to(device)
    model2 = MyModel3Layer(28*28, 12, 80)
    model2.to(device)
    step = 1
    loss_history1 = []
    loss_history2 = []
    for epoch in range(epochs):
        print(f'======= epoch {epoch+1} =======')
        for x, y in tqdm(train_loader):
            # 加载到GPU
            x = x.view((len(y), -1)).to(device)
            y = y.to(device)
            # 1层模型
            # 前向
            logits = model1.forward(x)
            # 损失
            loss, grad = myMSE(logits, y)
            loss_history1.append(loss.cpu())
            # 后向
            model1.backward(grad)
            # 更新参数
            model1.update(learning_rate)

            # 3层模型
            # 前向
            logits = model2.forward(x)
            # 损失
            loss, grad = myMSE(logits, y)
            loss_history2.append(loss.cpu())
            # 后向
            model2.backward(grad)
            # 更新参数
            model2.update(learning_rate)
            step += 1

    plt.plot(torch.arange(1, step), loss_history1, label='1 Layer Model')
    plt.plot(torch.arange(1, step), loss_history2, label='3 Layer Model')
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('loss.png')