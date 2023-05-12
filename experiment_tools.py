
#Config
from pathlib import Path
import os
import json
data_path = 'data'

from typing import List, Dict, Union, Tuple, Any
from functools import partial
import numpy as np

#Data
from torch.utils.data import Dataset, DataLoader # type: ignore
from PIL import Image

#Modeling
import torch # type: ignore


###### DATA PROCESSING ########


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
    def load_json(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    #Transform data series into appropriate format
    @staticmethod
    def transform_data_series(annotations: dict):
        chart_type = annotations['chart-type']
        data_series = annotations['data-series']
        x_series = [v['x'] for v in data_series]
        y_series = [v['y'] for v in data_series]
        return chart_type, x_series, y_series
    
    @staticmethod
    def s2n(x):

        try:
            x = float(x)

        except:
            x = x

        return x
    
    @staticmethod
    def round_float(value: Union[int, float, str]) -> Union[str, float]:
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
    def textify(chart_type, x_series, y_series, task = 'full_output'):
        x_series = 'x axis: ' + ', '.join([DataTools.round_float(x) for x in x_series])
        y_series = 'y axis: ' + ', '.join([DataTools.round_float(y) for y in y_series])
        chart_type = 'Chart type: ' + ' '.join(chart_type.split('_'))
        full_output = chart_type + ' ' + x_series + ' ' + y_series


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
    def numerify(answer: str):
        series = answer.split(';')
        series = [DataTools.s2n(x) for x in series]
        return series
    
    @staticmethod
    def get_image(img_path, b_w = True):
        
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
    
    
    @staticmethod
    def pix2struct_collator(processor, batch):
        """Colator for the 'pix2struct base' model
        """
        new_batch = {"flattened_patches":[], "attention_mask":[]}
        texts = [item["text"] for item in batch]
        
        text_inputs = processor(text=texts, 
                                padding=True, 
                                return_tensors="pt", 
                                add_special_tokens=True, 
                                truncation=True,
                                max_length=200)
        
        new_batch["labels"] = text_inputs.input_ids
        
        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
        
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])


        return new_batch
    
    @staticmethod
    def blip_collator(processor, batch):
        """Colator for the 'blip2-opt-2.7b' model
        """
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch


class BenetechDataset(Dataset):
    def __init__(self, 
                 processor,
                 dataset,
                 data_config: DataConfig, 
                 model_architecture: str = 'ybelkada/pix2struct-base',
                 task: str = 'full_output', 
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
        #identify the datapoint          
        item = self.dataset[idx]

        #get text from annotations file
        annotations = DataTools.load_json(item["annotations_path"])
        chart_type, x_series, y_series = DataTools.transform_data_series(annotations)
        text = DataTools.textify(chart_type, x_series, y_series, task=self.task)

        #Read image in grayscale from annotations image path
        img = Image.open(item["img_path"]).convert('L')
        encoding = self.processor(images= img, 
                                  return_tensors="pt", 
                                  add_special_tokens=True, 
                                  max_patches=self.data_config.max_patches)
        

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = text

        return encoding
    


##### MODELING #####

class CaptionGenerator:
    def __init__(self, 
                 model_architecture: str = "ybelkada/pix2struct-base",
                 lora_config: dict = None,
                 model_config: dict = None,
                 device:str = 'cuda',
                 optimizer_info:dict = {'name':'Adafactor',
                                        'optimizer_params':{'scale_parameter':False, 
                                                            'relative_step':False, 
                                                            'lr':0.01, 
                                                            'weight_decay':1e-05}
                                        },

                 source: str = 'pretrained',
                 task: str = 'full_output',
                 load_in_8bit: bool = True):
        
        assert model_architecture == 'ybelkada/pix2struct-base', 'Invalid model architecture'

        self.device = device
        self.task = task
        self.model_architecture = model_architecture
        self.source = source
        self.load_in_8bit = load_in_8bit
        self.optimizer_info = optimizer_info
        self.optimizer = None
        self.collator = DataTools.pix2struct_collator

        if lora_config is not None:
            from peft import LoraConfig
            self.config = LoraConfig(**lora_config)

        if model_config is not None:
            self.config = model_config

    def load_model(self, use_peft: bool = False):
        from transformers import AutoProcessor

        if (self.source == 'pretrained') & (self.model_architecture == "Salesforce/blip2-opt-2.7b"):
            pass

        elif (self.source == 'pretrained') & (self.model_architecture == "ybelkada/pix2struct-base"):
            from transformers import Pix2StructForConditionalGeneration
            from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup 

            model = Pix2StructForConditionalGeneration.from_pretrained(self.model_architecture).to(self.device)
            processor = AutoProcessor.from_pretrained(self.model_architecture)

        if use_peft:
            from peft import get_peft_model
            self.model = get_peft_model(model, self.config)
            self.model.print_trainable_parameters()


        else:
            self.model = model

        self.processor = processor

        if self.optimizer_info['name'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(model.parameters(), 
                                               lr=self.optimizer_info['optimizer_params']['lr'])

        elif self.optimizer_info['name'] == 'Adafactor':
            from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
            self.optimizer = Adafactor(model.parameters(), **self.optimizer_info['optimizer_params'])

            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                             num_warmup_steps=1000, 
                                                             num_training_steps=40000)


class ModelExperiment:
    def __init__(self, 
                 experiment_name: str, 
                 data_config: DataConfig, 
                 use_wandb: bool = False):

        self.experiment_name = experiment_name
        self.data_config = data_config
        self.use_wandb = use_wandb
        self.training_outputs = None

    def train_model(self, 
                    generator: CaptionGenerator, 
                    train_dataset: list, 
                    epochs: int = 10,
                    batch_size: int = 2):
        
        if generator.model is None:
            generator.load_model()
        
        generator.model.train()

        train_dataset = BenetechDataset(processor = generator.processor,
                                        dataset = train_dataset,
                                        data_config = self.data_config,
                                        task = generator.task,
                                        stage = 'train'
                                        )
        
        train_dataloader = DataLoader(train_dataset, 
                                      shuffle=True, 
                                      batch_size=batch_size, 
                                      collate_fn=partial(generator.collator, generator.processor))
                
        for epoch in range(epochs):
            print("Epoch:", epoch)
            for idx, batch in enumerate(train_dataloader):
                labels = batch.pop("labels").to(generator.device)
                flattened_patches = batch.pop("flattened_patches").to(generator.device)
                attention_mask = batch.pop("attention_mask").to(generator.device)

                outputs = generator.model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)
                
                loss = outputs.loss

                print("Loss:", loss.item())

                loss.backward()

                generator.optimizer.step()
                generator.optimizer.zero_grad()

                if (epoch + 1) % 20 == 0:
                    generator.model.eval()

                    predictions = generator.model.generate(flattened_patches=flattened_patches, 
                                                           attention_mask=attention_mask)        
                    print("Predictions:", generator.processor.batch_decode(predictions, 
                                                                           skip_special_tokens=True))

                    generator.model.train()

        return
        
    
    @staticmethod
    def evaluate_predictions():
        return

    