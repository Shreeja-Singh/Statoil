import json
import numpy as np
from tensorflow.python.lib.io import file_io

def preprocess(file):
    
    np.random.seed(seed = 42)
    json_data = json.load(open(file))

    images = []
    labels = []
    
    for data in json_data:
        
        img = np.ndarray((75,75, 2))
        
        band_1 = np.reshape(np.array(data['band_1']), (75, 75))
        band_2 = np.reshape(np.array(data['band_2']), (75, 75))
        
        mean_1 = np.mean(band_1)
        mean_2 = np.mean(band_2)
        
        std_1 = np.std(band_1)
        std_2 = np.std(band_2)
        
        diff_1 = band_1 - mean_1
        diff_2 = band_2 - mean_2
        
        norm_1 = diff_1 / std_1
        norm_2 = diff_2 /std_2
        
        img[:, :, 0] = norm_1
        img[:, :, 1] = norm_2
        
        images.append(img)
        labels.append([data['is_iceberg']])

    np.random.shuffle(images)
    np.random.shuffle(labels) 
       
    return images, labels
 
class batch():
    
    def __init__(self, images, labels):

        self.images = images
        self.labels = labels
        
    def calculate_index(self, mode, step):
        
        if mode == 'train':
            
            part = step % 12
            self.start = part * 100
            self.end = self.start + 100
            
            
        elif mode == 'eval':
            
            self.start = 1200
            self.end =  1400
            
            
        elif mode == 'test':     
           
            self.start = 1400
            self.end =  1604
            
        else:
            
           raise Exception('Mode can only take 3 values : train, eval & test')
            
    def batches(self, mode, step= 0):
        
        self.calculate_index(mode, step)
        
        images_final = np.array(self.images[self.start:self.end], dtype = np.float32)
        labels_final = np.array(self.labels[self.start:self.end])
       
        return images_final, labels_final 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
