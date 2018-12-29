import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """ 生成されたに対してInception Scoreを計算

    imgs -- (3x高さx幅) numpy画像のtorchデータセット
    cuda -- GPUを使うか否か
    batch_size -- inceptionモデルに与える際のバッチサイズ
    splits -- 分割数
    """
    #  画像の枚数　ーー＞　N
    N = len(imgs)

    #  画像枚数 N が バッチサイズ batch_size　以下ならアラート
    assert batch_size > 0
    assert N > batch_size

    #  入力画像がcudaなら　dtypeをcudaとする
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    #  入力画像群（imgs） を Dataloaderでバッチサイズだけ読み込み
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    #  学習済みのInception modelを読み込み
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    # 入力x をInception model に通し出力を得る関数
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # 実際に計算
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # KLダイバージェンスの平均の算出
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            #  scores: KLダイバージェンス
            scores.append(entropy(pyx, py))
        #  Inception Score 
        split_scores.append(np.exp(np.mean(scores)))
    #  Inception Score を　画像で平均、 分散を出力
    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))
