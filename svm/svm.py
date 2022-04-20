import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def predict(x, theta):
    return torch.argmax(theta @ x.view((-1,)))

def sgd(train_loader, test_loader, l, total_step):
    # initialize
    step = 1
    epoch = 1
    m = 28*28
    k = 10
    theta = torch.randn((k, m))

    # SGD loop
    acc = []
    while(step <= total_step):
        for x, y in train_loader:
            # to vector
            x = x.view((-1,))
            # the previous prediction
            c_hat = predict(x, theta)
            # the previous gradient
            grad = torch.zeros_like(theta)
            if c_hat != y:
                grad[c_hat] = x
                grad[y] = -x

            # update
            theta = (theta - 1/step * grad) / (1 - (2*l/step) / torch.sum(theta**2, dim=0)**0.5)

            # evaluation
            if step % 1000 == 0:
                preds, labels = [], []
                for x, y in test_loader:
                    preds.append(predict(x, theta))
                    labels.append(y[0])
                acc.append(accuracy_score(labels, preds))
            step += 1
        epoch += 1

    return theta, acc

if __name__ == '__main__':
    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train = True, download = True,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0, 1)
                        ])),
                        batch_size = 1, shuffle = True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train = False, download = True,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0, 1)
                        ])),
                        batch_size = 1, shuffle = False)
    # training
    df = pd.DataFrame()
    for l in [10, 1, 0.1, 0.01]:
        theta, acc = sgd(train_loader, test_loader, 1, 1e6)
        df[f'lambda={l}'] = acc
        # black and white figure
        theta = torch.sum(theta.abs(), dim=0).view((28, 28))
        theta = theta / (torch.sum(theta.abs()))
        print(theta)
        plt.imsave(f'lambda={l}.png', theta<1/(28*28), cmap='gray')

    # plot
    sns.lineplot(data=df)
    plt.ylabel('accuracy')
    plt.savefig('accuracy.png')
