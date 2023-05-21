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
from torch.utils.data import DataLoader
import multiprocessing
     


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
        self.train_img_paths = list(self.train_img_folder.iterdir())
        self.train_img_ids = [os.path.splitext(os.path.basename(img_path))[0] for img_path in self.train_img_paths]
        self.train_img_ids.sort()
        self.train_annotations_paths = list(self.train_annotations.iterdir())
        self.test_img_paths = list(self.test_img_folder.iterdir())
        self.test_img_ids = [os.path.splitext(os.path.basename(img_path))[0] for img_path in self.test_img_paths]
        self.test_img_ids.sort()

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
    def string2number(x):

        try:
            x = float(x)

        except:
            x = x

        return x
    @staticmethod
    def is_nan(value: Union[int, float, str]) -> bool:
        """
        Check if a value is NaN (not a number).

        Args:
            value (int, float, str): The value to check

        Returns:
            bool: True if the value is NaN, False otherwise
        """
        return isinstance(value, float) and str(value) == "nan"
    
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
    
    @staticmethod
    def get_gt_sequence(annotations_path: Union[str, os.PathLike]) -> Dict[str, str]:
        annotations = DataTools.load_json(annotations_path)

        PROMPT_TOKEN = "<|PROMPT|>"
        X_START = "<x_start>"
        X_END = "<x_end>"
        Y_START = "<y_start>"
        Y_END = "<y_end>"

        SEPARATOR_TOKENS = [PROMPT_TOKEN,
                                 X_START,
                                 X_END,
                                 Y_START,
                                 Y_END]

        LINE_TOKEN =  "<line>" 
        VERTICAL_BAR_TOKEN = "<vertical_bar>"
        HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
        SCATTER_TOKEN = "<scatter>"
        DOT_TOKEN = "<dot>"

        CHART_TYPE_TOKENS = [LINE_TOKEN,
                                  VERTICAL_BAR_TOKEN,
                                  HORIZONTAL_BAR_TOKEN,
                                  SCATTER_TOKEN,
                                  DOT_TOKEN]

        new_tokens = SEPARATOR_TOKENS + CHART_TYPE_TOKENS

        chart_type, x_series, y_series = DataTools.transform_data_series(annotations)

        all_x = []
        all_y = []
        for x,y in zip(x_series, y_series):
            x = DataTools.round_float(x)
            y = DataTools.round_float(y)

            if DataTools.is_nan(x) or DataTools.is_nan(y):
                continue
            all_x.append(x)
            all_y.append(y)
        
        chart_type_str = f"<{chart_type}>"
        x_series = X_START + ";".join(list(map(str, all_x))) + X_END
        y_series = Y_START + ";".join(list(map(str, all_y))) + Y_END
    
        gt_string = PROMPT_TOKEN + chart_type_str + x_series + y_series

        #return {
        #        "ground_truth": gt_string,
        #        "x": json.dumps(all_x),
        #        "y": json.dumps(all_y),
        #        "chart-type": chart_type,
        #        }

        return gt_string

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


            #train_dataset = dict({'img_path':[], 'annotations_path':[]})
            #val_dataset = dict({'img_path':[], 'annotations_path':[]})
            train_dataset = []
            val_dataset = []

            #Get train dataset
            for i in range(0,len(train_ids)):
                #train_dataset['img_path'].append(train_img_paths[i])
                #train_dataset['annotations_path'].append(train_annotations_paths[i])
                datapoint = {'img_path':train_img_paths[i], 
                             'annotations':DataTools.get_gt_sequence(train_annotations_paths[i])}
                train_dataset.append(datapoint)

            if len(val_ids) > 0:
                #Get val dataset
                for i in range(0,len(val_ids)):
                    #datapoint = dict({'img_path':[], 'annotations_path':[]})
                    #datapoint['img_path'].append(val_img_paths[i])
                    #datapoint['annotations_path'].append(val_annotations_paths[i])
                    datapoint = {'img_path':val_img_paths[i], 
                                 'annotations':DataTools.get_gt_sequence(val_annotations_paths[i])}
                    val_dataset.append(datapoint)

            dataset = {'train':train_dataset, 'val':val_dataset}

        elif stage == 'test':
            test_ids = self.data_config.test_img_ids
            test_img_paths = [self.data_config.test_img_folder/f'{img_id}.jpg' for img_id in test_ids]
            test_dataset = []

            #Get test dataset
            for i in range(0,len(test_ids)):
                test_dataset.append({'img_path':test_img_paths[i]})

            dataset = {'test':test_dataset}

        else:
            raise ValueError('Stage must be either dev or test')
        
        
        return dataset
    
    
    @staticmethod
    def pix2struct_collator(processor, batch):
        """Colator for the 'pix2struct base' model
        """
        new_batch = {"flattened_patches":[], "attention_mask":[]}
        texts = [item["labels"] for item in batch]
        
        
        text_inputs = processor(text=texts, 
                                padding="max_length", 
                                return_tensors="pt", 
                                add_special_tokens=True, 
                                truncation=True,
                                max_length=512)
        
        new_batch["labels"] = text_inputs["input_ids"]
        
        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
        
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])


        return new_batch


class BenetechDataset(Dataset):
    def __init__(self, 
                 processor,
                 dataset,
                 task: str = 'full_output',
                 max_patches: int = 1024,
                 max_length: int = 512,
                 split: str = "train",
                 ignore_id: int = -100):
        
        super().__init__()

        if task not in ['classify', 'extract_x', 'extract_y', 'full_output']:
            raise ValueError('Task must be either classify, extract_x, extract_y or full_output')
        
        self.task = task
        self.processor = processor
        self.dataset = dataset
        self.max_patches = max_patches
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):  
        #identify the datapoint          
        item = self.dataset[idx]

        #get text from annotations file
        #annotations = DataTools.load_json(item["annotations_path"])
        #chart_type, x_series, y_series = DataTools.transform_data_series(annotations)
        #text = DataTools.textify(chart_type, x_series, y_series, task=self.task)
        #gt_sequence = DataTools.get_gt_sequence(item["annotations_path"])
        gt_sequence = item["annotations"]

        #Read image in grayscale from annotations image path
        #img = Image.open(item["img_path"]).convert('L')
        img = DataTools.get_image(item["img_path"])

        encoding = self.processor(images= img, 
                                  return_tensors="pt", 
                                  max_patches=self.max_patches)
        

        encoding = {k:v.squeeze() for k,v in encoding.items()}
        #encoding["text"] = text
        #input_ids = self.processor.tokenizer(gt_sequence,
        #                                     max_length=self.max_length,
        #                                     padding="max_length",
        #                                     truncation=True,
        #                                     return_tensors="pt").input_ids
        
        #labels = input_ids.squeeze().clone()
        #labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        encoding["labels"] = gt_sequence #.squeeze()
        #labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return encoding

##### MODELING #####

class CaptionGenerator:
    def __init__(self, 
                 model_architecture: str = "google/pix2struct-base",
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
                 load_in_8bit: bool = False):
        
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

    def load_model(self, use_peft: bool = False, additional_tokens: list = []):
        #if (self.source == 'pretrained') & (self.model_architecture == "Salesforce/blip2-opt-2.7b"):
        #    pass
        
        self.added_tokens = []
        p2sbase = self.model_architecture == "google/pix2struct-base"
        if (self.source == 'pretrained') & (p2sbase):
            from transformers import AutoProcessor
            from transformers import Pix2StructForConditionalGeneration
            from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup 

            model = Pix2StructForConditionalGeneration.from_pretrained(self.model_architecture,
                                                                       is_encoder_decoder=True).to(self.device)
            processor = AutoProcessor.from_pretrained(self.model_architecture)

        if use_peft:
            from peft import get_peft_model
            self.model = get_peft_model(model, self.config)
            self.model.print_trainable_parameters()


        else:
            self.model = model

        if len(additional_tokens) > 0:
            newly_added_num = processor.tokenizer.add_tokens(additional_tokens)
            if newly_added_num > 0:
                    model.decoder.resize_token_embeddings(len(processor.tokenizer))
                    self.added_tokens.extend(additional_tokens)

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
                    batch_size: int = 4,
                    NUM_ACCUMULATION_STEPS: int = 8):
        
        train_dataset = BenetechDataset(processor = generator.processor,
                                        dataset = train_dataset,
                                        task = generator.task,
                                        )
        
        train_dataloader = DataLoader(train_dataset, 
                                      shuffle=True, 
                                      batch_size=batch_size, 
                                      collate_fn=partial(generator.collator, generator.processor))
        
        
        generator.model.train()
                
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
                loss = loss / NUM_ACCUMULATION_STEPS #for gradient accumulation

                print("Loss:", loss.item())

                loss.backward()

                if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_dataloader)):
                    generator.optimizer.zero_grad()
                    # Update Optimizer
                    generator.optimizer.step()


                #generator.optimizer.step()
                #generator.optimizer.zero_grad()

                if (epoch + 1) % 5 == 0:
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