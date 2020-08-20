import os.path as osp
import glob
from PIL import Image
import torch.utils.data as data

from utils.data_augumentation import Compose, Resize_Totensor

class make_datapath_list():
  def __init__(self, rootpath):
    """
    rootpath: Absolute path
    """
    img_file_path = sorted(glob.glob(rootpath+ '/kaggle_3m/TCGA*'))
    self.train_file_path = img_file_path[:75]
    self.val_file_path = img_file_path[75:95]
    self.test_file_path = img_file_path[95:]
  
  def get_list(self, path_type):
    """
    path_type: select a path type from "train", "val" and "test"
    """
    if path_type=="train":
      file_path = self.train_file_path

    elif path_type=="val":
      file_path = self.val_file_path

    else:
      file_path = self.test_file_path

    img_list = []
    anno_list = []
    for path in file_path:
      path = glob.glob(path+"/*.tif")
      img_path = sorted([p for p in path if "mask" not in p])
      anno_path = [p[:-4]+"_mask.tif" for p in img_path]
      img_list += img_path
      anno_list += anno_path

    return img_list, anno_list

class DataTransform():
    """
    画像のサイズをinput_size x input_sizeにする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    """

    def __init__(self, input_size):
        self.data_transform = {
            'train': Compose([
                Resize_Totensor(input_size)
            ]),
            'val': Compose([
                Resize_Totensor(input_size)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


class BrainDataset(data.Dataset):
    """
    脳MRIのDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img