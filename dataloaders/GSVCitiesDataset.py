# https://github.com/amaralibey/gsv-cities
import os

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229]),
])


class GSVCitiesDataset(Dataset):
    def __init__(self,
                 img_per_place=5,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 transform_e=default_transform,
                 train_anno=None,
                 base_path=None,
                 ):
        super(GSVCitiesDataset, self).__init__()
        assert base_path is not None, 'you have to provide the base_path of data set'
        assert train_anno is not None, 'you have to provide the train list'
        self.base_path = base_path

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.transform_e = transform_e
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes(train_anno)
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self, train_anno):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(train_anno)
        df = df.sample(frac=1)  # shuffle the city dataframe
        if 'panoid_e' in df.columns:
            self.bimod = True
        else:
            self.bimod = False
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        img_e_s = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = os.path.join(self.base_path, img_name)
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)
            
            if self.bimod:
                img_name = self.get_img_name(row, key='panoid_e')
                img_path = os.path.join(self.base_path, img_name)
                img_e = self.image_loader(img_path)

                if self.transform_e is not None:
                    img_e = self.transform_e(img_e)
                img_e_s.append(img_e)
            imgs.append(img)

        if not self.bimod:
            return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)
        else:
            return torch.stack(imgs), torch.stack(img_e_s), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('L')

    @staticmethod
    def get_img_name(row, key='panoid'):
        panoid = row[key]
        name = panoid
        return name


class GSVCitiesValDataset(Dataset):
    def __init__(self,
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=None,
                 query_anno=None,
                 ref_anno=None,
                 key=None):
        super(GSVCitiesValDataset, self).__init__()
        assert base_path is not None, 'you have to provide the base_path of data set'
        self.base_path = base_path

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        if key is None:
            self.key = 'panoid'
        else:
            self.key = key 
        self.dataframe = self.__getdataframes(query_anno, ref_anno)
        self.label = self.dataframe['place_id']
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
        
    def __getdataframes(self, query_anno, ref_anno):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df_db = pd.read_csv(query_anno)
        df_ref = pd.read_csv(ref_anno)
        self.num_references = len(df_ref)
        concatenated_df = pd.concat([df_ref, df_db], ignore_index=True, sort=False)
        return concatenated_df
    
    def __getitem__(self, index):
        
        # get the place in form of a dataframe (each row corresponds to one image)
        row = self.dataframe.loc[index]
        img_name = self.get_img_name(row, self.key)
        img_path = os.path.join(self.base_path, img_name)
        img = self.image_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.dataframe)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('L')

    @staticmethod
    def get_img_name(row, key='panoid'):
        panoid = row[key]
        name = panoid
        return name