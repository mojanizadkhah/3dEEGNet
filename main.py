import os
import copy
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, StratifiedGroupKFold

# Custom utility functions
from utils import (
    direction, evaluate, threedconversion, DatasetLoader, 
    train_model, adhoc_augmentation_full_dataset, filter_extreme_voxelwise
)

name = 'test_sal'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

address = '/Users/mojanizadkhah/Desktop'

save_dir = f'{address}/{name}/'




num_epochs = 10
normalize = False
training_mode = True
threshold = 0.6
batch_size = 15

participants = list(range(1,21))

p = 25
fc_features = p*48

num_trials = 1
max_restarts = 1
num_epochs = 1

increase_factor_train = num_epochs


for stim in ['saccade', 'stimulus']: 
    for task in ['SA', 'RL_ANTI', 'RL_PRO', 'R', 'L', 'RR', 'LL']:  
        for par in participants:
            base_dir = f'{address}/pre-proccessed/{p}datapoints/{stim}/{task}/1_epochs/{par}'

            DIR = f"{base_dir}/"
            print("Processing participant", par)

            # Load data for the participant
            X, y = DatasetLoader(DIR)
            X = threedconversion(X)

            # X, y, num_removed = filter_extreme_voxelwise(X, y, std_threshold=5)
            # print(f"Removed {num_removed} extreme trials.")

            X = np.expand_dims(X, axis=1).astype('float32')
            y = y.astype('float32').reshape(-1)

            fold_metrics_df = train_model(training_mode, X, y, num_epochs, p, fc_features, max_restarts, par, stim, task, save_dir, evaluate, 
                                        adhoc_augmentation_full_dataset, num_trials, increase_factor_train, batch_size, normalize)
                                        
            if(training_mode):
                df_results = pd.DataFrame(fold_metrics_df, columns=['Iteration', 'Last Epoch Train ACC', 'Last Epoch Train AUC', 'Last Epoch Val ACC', 'Last Epoch Val AUC'])
            del X, y
            gc.collect()
            torch.cuda.empty_cache()
