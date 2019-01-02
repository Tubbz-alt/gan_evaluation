# coding: utf-8

#### WGAN は GANに置ける勾配消失問題の解決策として提案された
#### 勾配消失問題は生成画像の分布と真の画像の分布が完全に識別器によって識別可能な時、損失が0となることから起こる
#### GANとの相違点　① 損失関数　交差エントロピー　-->> Earth mover distance, 
#### GANとの相違点　② 識別器の構造

#### GAN とは　JSダイバージェンスという2つの確率密度間の距離を表す指標を学習によって最小化する
#### JSダイバージェンス　勾配消失問題の原因　（生成モデルのパラメータの最適地周りで勾配が「０」になる）
#### Earth Mover Distance では上のような状況において勾配消失が起きない　安定した学習が可能

#### 実際の学習手順
#### 学習画像のミニバッチを識別モデルに通した出力の平均  f(x),   生成画像のミニバッチを識別モデルに通した出力の平均　f(x^)
#### ①   f(x) - f(x^)を最大化するように識別モデルのパラメータ　θ_d　を更新
#### ②   θ_d <<-- clip(θ_d, -c,c)  c=0.01程度　　識別モデルの重みを -c ~ c の範囲に限定する
#### ③   生成モデルのパラメータ　θ_g　を f(x^)（生成画像を識別モデルに入力した時に訓練画像と判別されるように）を更新
#### ①〜③を繰り返す

#### f(x) - f(x^) ：Earth Mover Distance を表す　　生成モデルはf(x^)を最大化することで、生成画像のベンぷを真の画像の分布に近づける

import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from inception_score import *
from fid_score import *


class generator(nn.Module):

    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            # nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class WGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.batch_scoring = args.batch_scoring
        self.batch_scoring_fid = args.batch_scoring_fid
        self.batch_size_pretraining = args.batch_pretraining
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62

        self.c = 0.01                   # clipping value
        self.n_critic = 5               # the number of iterations of the critic per generator iteration

        self.alpha = 0.8
        self.num_sigma = 5
        self.opt_preite = args.opt_preite
        self.fid_score = 20

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.data_loader_scoring = dataloader(self.dataset, self.input_size, self.batch_scoring)
        self.data_loader_scoring_fid = dataloader(self.dataset, self.input_size, self.batch_scoring_fid)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.train_hist['FID'] = []
        self.train_hist['inception_score'] = []


        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)
                D_loss = D_real_loss + D_fake_loss

                D_loss.backward()
                self.D_optimizer.step()

                # clipping D
                for p in self.D.parameters():
                    p.data.clamp_(-self.c, self.c)

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
# inception_score.py
                    self.G.eval()

                    num_of_scoring = int(50000 / self.batch_scoring)
                    if ((iter+1)%500) == 0:
                        with torch.no_grad():
                            IS = []
                            for nsplits in range(num_of_scoring):
                                G_for_scoring = self.G(torch.rand(self.batch_scoring, self.z_dim).cuda())
                                IS_mean, IS_std = inception_score(G_for_scoring, cuda=True, batch_size=32, resize=True, splits=10)
                                IS = np.append(IS, IS_mean)
                            self.train_hist['inception_score'].append(IS_mean)
                            print("I_Score", IS_mean)
# fid_score.py

                        FID =[]
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
                                    print("FID.shape", FID.shape)
                                self.train_hist['FID'].append(np.mean(FID))
                                print("FID", np.mean(FID))

                    self.G.train()    
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.item())

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
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.D_loss_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.G_loss_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

        utils.score_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.score_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

        utils.fid_save(self.train_hist,os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.fid_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

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
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '_FID_score' +'.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_FID_score.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '__FID_score.pkl'))

        with open(os.path.join(save_dir, self.model_name + 'FID_score.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_FID_score.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_FID_score.pkl')))
