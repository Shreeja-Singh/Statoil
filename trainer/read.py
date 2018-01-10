import json
import sys
import numpy as np


def preprocess(file):
    
    np.random.seed(seed = 42)
    data = json.load(open(file))

    images = []
    labels = []
    
    for data in json_data:
        
        img = np.ndarray((75,75, 2))
        
        band_1 = np.reshape(np.array(data['band_1']), (75, 75))
        band_2 = np.reshape(np.array(data['band_2']), (75, 75))
        
        img[:, :, 0] = band_1
        img[:, :, 1] = band_2
        
        images.append(img)
        labels.append([data['is_iceberg']])

    np.random.shuffle(images)
    np.random.shuffle(labels) 
       
    return images, labels
 
class batch():
    
    def __init__(self, step, images, labels, mode = 'train'):

        self.mode = 'train'
        self.step = step
        self.images = images
        self.labels = labels
        
    def calculate_index(self):
        
        if self.mode == 'train':
            
            part = self.step % 12
            self.start = (part - 1) * 100
            self.end = self.start + 99
            
        elif self.mode == 'eval':
            
            self.start = 1200
            self.end =  1399
            
        elif self.mode == 'test':     
           
            self.start = 1400
            self.end =  1604
            
    def batches(self):
        
        self.calculate_index()

       
        return np.array(images[self.start:self.end]), np.array(labels[self.start:self.end])
        
           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
