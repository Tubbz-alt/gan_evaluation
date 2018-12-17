# coding: utf-8
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
############### 追記箇所　以下#############################################################################################
from inception_score import *    
############### 追記箇所　以上#############################################################################################

############### 追記箇所　以下#############################################################################################
from fid_score import *
############### 追記箇所　以上#############################################################################################

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        #  ノイズの次元数
        self.input_dim = input_dim
        #  出力画像f-img のチャネル数
        self.output_dim = output_dim
        #  出力画像の縦横の大きさ
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        #  ノイズ100次元　→ 128×8×8 → (64, 8, 8) → (64, 16, 16) → (1, 32, 32) 生成画像 f-img
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        # 重み（畳み込み、転地畳み込み、全結合）を「平均0, 標準偏差0.02」の正規分布から初期化
        utils.initialize_weights(self)

    def forward(self, input):
        # ノイズを入力として全結合
        x = self.fc(input)
        # 出力のベクトル（128×8×8）を テンソル（1, 128, 8, 8）に変換
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        # テンソルを転地畳み込みで拡大して、f-img(生成画像)を出力
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        #  入力画像のチャネル数　             {1 -->> グレースケール, 3 -->> カラー画像}
        self.input_dim = input_dim
        #  入力画像が生成画像か訓練画像かの確率　{1  -->> 訓練画像,　   0 -->> 生成画像}
        self.output_dim = output_dim
        #  入力画像の縦横
        self.input_size = input_size

        #  {1, 32, 32}の画像を入力　-->> {128, 8, 8}出力
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        #  {128, 8, 8}入力　を {128×8×8}にして　-->> 1024 -->> 1 (確率)
        #self.fc = nn.Sequential(
        #    nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
        #    nn.BatchNorm1d(1024),
        #  {128, 8, 8}入力　を {128×8×8}にして　-->> 1024 -->> 1 (確率)
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        # 重み（畳み込み、転地畳み込み、全結合）を「平均0, 標準偏差0.02」の正規分布から初期化
        utils.initialize_weights(self)

    def forward(self, input):

        #  入力画像を畳み込み
        x = self.conv(input)
        #  出力の画像{128, 8, 8} -->> {1, 128×8×8}
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        #  {1, 128×8×8} -->> 1 全結合
        x = self.fc(x)

        return x

class GAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        #  可視化する生成画像の枚数
        self.sample_num = 100
        #  バッチサイズ
        self.batch_size = args.batch_size
        self.batch_scoring = args.batch_scoring
        self.batch_scoring_fid = args.batch_scoring_fid
        self.batch_pretraining = args.batch_pretraining
        self.save_dir = args.save_dir
        #  結果を残すファイル reslut_dirに残す
        self.result_dir = args.result_dir
        #  GANを実行するデータセット
        self.dataset = args.dataset
        #  結果をlogとして残すディレクトリの名前
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        #  GANの種類を定義
        self.model_name = args.gan_type
        self.input_size = args.input_size
        # 生成モデルの入力のノイズの次元数
        self.z_dim = 30
        #self.z_dim = self.batch_pretraining * (2 * self.batch_pretraining + 1)
############### UTの追記箇所　以下#############################################################################################
        self.alpha = 0.8
        #self.num_sigma = 5
        self.fid_score = 20
############### UTの追記箇所　以下#############################################################################################
        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.data_loader_scoring = dataloader(self.dataset, self.input_size, self.batch_scoring)
        self.data_loader_scoring_fid = dataloader(self.dataset, self.input_size, self.batch_scoring_fid)
        self.data_loader_pretraining = dataloader(self.dataset, self.input_size, self.batch_pretraining)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        #  生成モデルと識別モデルの初期化　ノイズの次元数、生成画像の縦横、出力画像のチャネル数を定義
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        #   生成モデル、識別モデルの最適化手法の定義　
        #   それぞれ更新するパラメータを定義することで生成モデルにおいて識別モデルのパラメータを更新しない
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

############### UTの追記箇所　以下#############################################################################################
        self.optimizerG2 = optim.Adam(self.G.parameters(), lr=args.lrG_FRECHET, betas=(args.beta1, args.beta2))
############### UTの追記箇所　以下#############################################################################################

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            #  交差エントロピーを損失関数（評価関数）として定義
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')


        # fixed noise
        #  生成モデルの更新の前にノイズを新しく生成し直す必要がある
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
############### 追記箇所　以下#############################################################################################
        self.train_hist['FID'] = []
        self.train_hist['inception_score'] = []        
############### 追記箇所　以上#############################################################################################

        #  訓練画像のラベル　-->> 1,  生成画像のラベル　-->> 0
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        #  GPUなら変数を.cuda()で定義
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        #  エポック数だけの更新
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            # iter : 訓練画像の番号（index）,     (x_, _) : 訓練画像
            # 最後のエポックがバッチサイズにみたない場合は無視
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                #  生成モデルの入力のノイズ
                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                #  識別モデルの更新
                self.D_optimizer.zero_grad()

                #  識別モデルにとって訓練画像の認識結果は「 1 」 に近いほどよい
                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                #  生成モデルにとって生成画像の認識結果は「 0 」 に近いほどよい
                G_ = self.G(z_)
                
############### 追記箇所　以下#############################################################################################
# inception_score.py (pytorch)
                self.G.eval()
                num_of_scoring = int(50000 / self.batch_scoring)
                if ((iter+1)%500) == 0:
                    with torch.no_grad():
                        IS = []
                        for nsplits in range(num_of_scoring):
                            G_for_scoring_inception = self.G(torch.rand(self.batch_scoring, self.z_dim).cuda())
                            IS_mean, IS_std = inception_score(G_for_scoring_inception, cuda=True, batch_size=32, resize=True, splits=1)
                            IS = np.append(IS, IS_mean)
                        self.train_hist['inception_score'].append(np.mean(IS))
                        print("I_Score", IS_mean)
# fid_score.py
                    FID = [] 
                    for iter_fid, (x_for_scoring, _) in enumerate(self.data_loader_scoring_fid):
                        x_for_scoring = x_for_scoring.cuda()
                        if ((iter_fid+1)%100)==0:
                            with torch.no_grad():
                                for preepoch in range(self.fid_score):
                                    G_for_scoring_frechet = self.G(torch.rand(self.batch_scoring_fid, self.z_dim).cuda())
                                    G_for_scoring_frechet = G_for_scoring_frechet / G_for_scoring_frechet.max()
                                    Incep_mean_z, Incep_std_z = calculate_activation_statistics(G_for_scoring_frechet)
                                    Incep_mean_x, Incep_std_x = calculate_activation_statistics(x_for_scoring)
                                    frechet_inception_distance = calculate_frechet_distance(Incep_mean_z, Incep_std_z, Incep_mean_x, Incep_std_x)
                                    FID = np.append(FID, frechet_inception_distance)
                            self.train_hist['FID'].append(np.mean(FID))
                            print("FID", np.mean(FID))
############### 追記箇所　以上#############################################################################################
                self.G.train()
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                #  2つのlossの和を最小化
                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                #  生成モデルのパラメータの更新を行わない
                self.D_optimizer.step()

                # 生成モデルの更新

                self.G_optimizer.zero_grad()

                #  生成モデルにとって生成画像の識別結果は　「 １ 」に近いほど良い
                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                #  識別モデルの更新を行わない
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        #   animationを保存
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        #  loss, score を 保存
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.D_loss_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.G_loss_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
 
        utils.score_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.score_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

        utils.fid_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.fid_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '_UT_pretrain_FID_score' +'.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G_UT_pretrain_FID_score.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D_UT_pretrain_FID_score.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history_UT_pretrain_FID_score.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G_UT_pretrain_FID_score.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D_UT_pretrain_FID_score.pkl')))
