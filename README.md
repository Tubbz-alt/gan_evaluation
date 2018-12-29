# gan_evaluation
GANの評価

学習済みのGANのモデルに対して、生成される画像の質や、多様性を評価する。
評価尺度はFrechet Inception Distance(FID)とInception Score(FID)


Frechet Inception Distance:https://arxiv.org/abs/1706.08500
Inception Score:http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans

ACGAN.py、GAN.pyにてモデルの構造、loss、学習ほうなどを定義
fid_score.pyでFIDの、inception_score.pyでISの計算法を定義
inception.pyではFIDやISを計算するときに用いる学習済みの分類モデル(Inceptionモデル)を定義


main.pyにてモデルの学習を実行、GAN.pyでは学習時にFID、ISを保存し可視化する
