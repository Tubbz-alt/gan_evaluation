import argparse, os, torch

from ACGAN import ACGAN


## 変数の定義 ##
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='ACGAN',
                        choices=['ACGAN'],
                        help='The type of GAN')

    parser.add_argument('--dataset', type=str, default='lsun', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--batch_scoring', type=int, default=5000, help='The size of batch for inception score')
    parser.add_argument('--batch_scoring_fid', type=int, default=500, help='The size of batch for frechet inception distance')
    parser.add_argument('--batch_pretraining', type=int, default=5000, help='The size of batch for pretraining sigma')
    parser.add_argument('--batch_ut', type=int, default=64, help='The size of batch for inception pretraining')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrG_FRECHET', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    parser.add_argument('--opt_preite', type=int, default=40, help='pretraining iteration number')
    return check_args(parser.parse_args())

## ディレクトリのチェック ##
def check_args(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

# main関数 #
def main():
    # ハイパーパラメータの設定
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    # GANのモデルを決定
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'ACGAN':
        gan = ACGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    gan.train()
    print(" [*] Training finished!")

    # 学習済みの生成器が生成するイメージの可視化
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")
    #gan.load_generated_images_for_scoring(50000)
    
if __name__ == '__main__':
    main()
