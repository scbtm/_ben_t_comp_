
#Config
from pathlib import Path
import os
import json
data_path = 'data'

from typing import List, Dict, Union, Tuple, Any

#Data
from torch.utils.data import Dataset, DataLoader # type: ignore
from PIL import Image

#Modeling
import torch # type: ignore





class DataConfig:
    def __init__(self, path):
        self.max_patches = 1024 #could be 2048 or 4096 for larger models
        self.path = Path(path)
        self.train = self.path/'train'
        self.test = self.path/'test'
        self.train_img_folder = self.train/'images'
        self.train_annotations = self.train/'annotations'
        self.test_img_folder = self.test/'images'
        #self.test_annotations = self.test/'annotations'
        self.train_img_paths = list(self.train_img_folder.iterdir())
        self.train_img_ids = [os.path.splitext(os.path.basename(img_path))[0] for img_path in self.train_img_paths]
        self.train_img_ids.sort()
        self.train_annotations_paths = list(self.train_annotations.iterdir())
        self.test_img_paths = list(self.test_img_folder.iterdir())
        self.test_img_ids = [os.path.splitext(os.path.basename(img_path))[0] for img_path in self.test_img_paths]
        self.test_img_ids.sort()
        #self.test_annotations_paths = list(self.test_annotations.iterdir())

class DataTools:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config

    #Load annotations
    @staticmethod
    def load_json(self,path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    #Transform data series into appropriate format
    @staticmethod
    def transform_data_series(self, annotations: dict):
        chart_type = annotations['chart_type']
        data_series = annotations['data_series']
        x_series = [v['x'] for v in data_series]
        y_series = [v['y'] for v in data_series]
        return chart_type, x_series, y_series
    
    @staticmethod
    def s2n(self, x):

        try:
            x = float(x)

        except:
            x = x

        return x
    
    @staticmethod
    def round_float(self,value: Union[int, float, str]) -> Union[str, float]:
        """
        Convert a float value to a string with the specified number of decimal places. 
        If there is more than 1 digit in the integer, then we will truncate to 1 decimal.
        Otherwise, will truncate to 4 decimals.

        Args:
            value (int, float, str): The float value to convert

        Returns:
            str: The rounded float value as a string
        """
        if isinstance(value, int|float):
            value = str(value)

            if "." in value:
                integer, decimal = value.split(".")
                if abs(float(integer)) > 1:
                    decimal = decimal[:2]
                else:
                    decimal = decimal[:4]

                value = integer + "." + decimal
            
        return value
    
    @staticmethod
    def textify(self,chart_type, x_series, y_series, task = 'classify'):
        x_series = 'x series: ' + ';'.join([self.round_float(x) for x in x_series])
        y_series = 'y series: ' + ';'.join([self.round_float(y) for y in y_series])
        chart_type = 'Chart type: ' + chart_type
        full_output = chart_type + '|' + x_series + '|' + y_series


        if task == 'classify':
            return chart_type
        
        elif task == 'extract_x':
            return x_series
        
        elif task == 'extract_y':
            return y_series
        
        elif task == 'full_output':
            return full_output
        
        else:
            raise ValueError('Task must be either classify, extract_x, extract_y or full_output')
        
    @staticmethod
    def numerify(self, answer: str):
        series = answer.split(';')
        series = [self.s2n(x) for x in series]
        return series
    
    @staticmethod
    def get_image(self, img_path, b_w = True):
        
        if b_w:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')

        return img

    def build_dataset(self, 
                      stage: str = 'dev', 
                      train_val_split: float = 0.8,
                      sample_size: int = None):
        
        if stage not in ['dev', 'test']:
            raise ValueError('Stage must be either dev or test')
        
        if stage == 'dev':
            if train_val_split < 1.0:

                #Split data into train and validation sets
                split_value = int(len(self.data_config.train_img_ids)*train_val_split)
                train_ids = self.data_config.train_img_ids[:split_value]
                val_ids = self.data_config.train_img_ids[split_value:]

                if sample_size is not None:
                    train_ids = train_ids[:sample_size]
                    val_ids = val_ids[:sample_size]

                #Get img and annotation ids from train and validation sets
                train_img_paths = [self.data_config.train_img_folder/f'{img_id}.jpg' for img_id in train_ids]
                val_img_paths = [self.data_config.train_img_folder/f'{img_id}.jpg' for img_id in val_ids]

                train_annotations_paths = [self.data_config.train_annotations/f'{img_id}.json' for img_id in train_ids]
                val_annotations_paths = [self.data_config.train_annotations/f'{img_id}.json' for img_id in val_ids]


            else:
                train_ids = self.data_config.train_img_ids
                val_ids = []

                if sample_size is not None:
                    train_ids = train_ids[:sample_size]

                train_img_paths = self.data_config.train_img_paths
                val_img_paths = []

                train_annotations_paths = self.data_config.train_annotations_paths
                val_annotations_paths = []


            train_dataset = []
            val_dataset = []

            #Get train dataset
            for i in range(0,len(train_ids)):
                img_id = train_ids[i]
                img_path = train_img_paths[i]
                annotations_path = train_annotations_paths[i]

                datapoint = {'img_id': img_id, 'img_path': img_path, 'annotations_path': annotations_path}
                train_dataset.append(datapoint)

            if len(val_ids) > 0:
                #Get val dataset
                for i in range(0,len(val_ids)):
                    img_id = val_ids[i]
                    img_path = val_img_paths[i]
                    annotations_path = val_annotations_paths[i]

                    datapoint = {'img_id': img_id, 'img_path': img_path, 'annotations_path': annotations_path}
                    val_dataset.append(datapoint)

            dataset = {'train': train_dataset, 'val': val_dataset, 'test': None}

        elif stage == 'test':
            test_ids = self.data_config.test_img_ids
            test_img_paths = [self.data_config.test_img_folder/f'{img_id}.jpg' for img_id in test_ids]
            test_dataset = []

            #Get test dataset
            for i in range(0,len(test_ids)):
                img_id = test_ids[i]
                img_path = test_img_paths[i]
                datapoint = {'img_id': img_id, 'img_path': img_path, 'annotations_path': None}
                test_dataset.append(datapoint)

            dataset = {'train': None, 'val': None, 'test': test_dataset}

        else:
            raise ValueError('Stage must be either dev or test')
        
        
        return dataset

class BenetechDataset(Dataset):
    def __init__(self, 
                 processor,
                 dataset,
                 data_config: DataConfig, 
                 task: str = 'classify', 
                 stage: str = 'train'):
        

        if task not in ['classify', 'extract_x', 'extract_y', 'full_output']:
            raise ValueError('Task must be either classify, extract_x, extract_y or full_output')
        if stage not in ['train', 'val', 'test']:
            raise ValueError('Stage must be either train, val or test')

        self.data_config = data_config
        self.task = task
        self.stage = stage
        self.processor = processor
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=Image.open(item["img_path"]), 
                                  return_tensors="pt", 
                                  add_special_tokens=True, 
                                  max_patches=self.data_config.max_patches)
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        annotations = self.load_json(item["annotations_path"])
        chart_type, x_series, y_series = self.transform_data_series(annotations)

        text = DataTools.textify(chart_type, x_series, y_series, task=self.task)

        encoding["text"] = text
        
        return encoding