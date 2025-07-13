import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGNet(nn.Module):
    def __init__(self, fc_features):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1: Temporal Convolution
        self.temporal_conv = nn.Conv3d(1, 24, (3, 1, 1), padding=(1, 0, 0))
        self.batchnorm1 = nn.BatchNorm3d(24, affine=False)
        
        # Layer 2: Depthwise Convolution
        self.depthwise_conv = nn.Conv3d(24, 24, (1, 3, 3), padding=(0, 1, 1), groups=24)
        self.batchnorm2 = nn.BatchNorm3d(24, affine=False)
        self.pooling2 = nn.MaxPool3d((1, 2, 2))

        # Layer 3: Separable Convolution
        self.separable_conv_depthwise = nn.Conv3d(24, 24, (1, 3, 3), padding=(0, 1, 1), groups=24)
        self.separable_conv_pointwise = nn.Conv3d(24, 12, (1, 1, 1))
        self.batchnorm3 = nn.BatchNorm3d(12, affine=False)
        self.pooling3 = nn.MaxPool3d((1, 2, 2))

        # Fully connected layer now outputs 2 values
        self.fc1 = nn.Linear(fc_features, 2)  

    def forward(self, x):
        batch_size = x.size(0)
        
        # Layer 1: Temporal Convolution
        x = F.elu(self.temporal_conv(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.15)
        
        # Layer 2: Depthwise Convolution
        x = F.elu(self.depthwise_conv(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        
        # Layer 3: Separable Convolution
        x = F.elu(self.separable_conv_depthwise(x))
        x = self.separable_conv_pointwise(x)
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        
        # Flatten before the fully connected layer
        x = x.view(batch_size, -1)
        x = self.fc1(x)  # Now outputs 2 values per sample
        
        # Option 1: Use softmax to get probabilities
        # x = F.softmax(x, dim=1) 

        # Option 2: Return raw values and let the user apply argmax
        return x  # Use torch.argmax(x, dim=1) to get the final prediction class

def direction(task,targett):
    if task =='RL_PRO':
        if targett==0:
            return 'Left_movement'
        elif targett==1:
            return 'Right_movement'
    if task =='RL_ANTI':
        if targett==0:
            return 'Right_movement'
        elif targett==1:
            return 'Left_movement'
    if task in ['SA', 'R','L','RR','LL']:
        if targett==0:
            return 'Pro'
        elif targett==1:
            return 'Anti'


def filter_extreme_voxelwise(X, y, std_threshold=5):
    """
    Removes trials from X and y where any voxel exceeds its voxelwise mean + threshold * std.
    
    Parameters:
    - X: np.ndarray, shape (n_trials, time, 9, 9)
    - y: np.ndarray, shape (n_trials,) or (n_trials, 1)
    - std_threshold: float, number of std deviations above mean per voxel
    
    Returns:
    - X_filtered: np.ndarray with filtered trials
    - y_filtered: np.ndarray with filtered labels
    - num_removed: int, number of trials removed
    """
    assert X.shape[0] == y.shape[0], "Mismatch in number of trials between X and y"
    
    # Compute voxel-wise mean and std across trials (axis=0)
    voxel_mean = np.mean(X, axis=0)  # shape: (time, 9, 9)
    voxel_std = np.std(X, axis=0)    # shape: (time, 9, 9)

    # Compute voxel-wise threshold
    threshold = voxel_mean + std_threshold * voxel_std  # shape: (time, 9, 9)

    # Create a mask of shape (n_trials,) where True means trial is clean
    is_clean_trial = np.all(X <= threshold, axis=(1, 2, 3)) & np.all(X >= -threshold, axis=(1, 2, 3))

    num_removed = np.sum(~is_clean_trial)

    # Filter X and y
    X_filtered = X[is_clean_trial]
    y_filtered = y[is_clean_trial]

    return X_filtered, y_filtered, num_removed



def evaluate(net, data_loader, params):
    net.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs) 
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy()) 

    all_outputs = np.concatenate(all_outputs, axis=0)  # Shape: (num_samples, 2)
    all_labels = np.concatenate(all_labels, axis=0)  # Shape: (num_samples,)

    acc = accuracy_score(all_labels, all_outputs.argmax(axis=1))

    auc = 0.0
    if len(np.unique(all_labels)) > 1:  # Ensure both classes exist
        probabilities = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()[:, 1]  # Get probability for class 1
        auc = roc_auc_score(all_labels, probabilities)

    return acc, auc

def threedconversion(Data):
    epoch_num, length_epoch_channel = Data.shape[0], Data.shape[1]
    channel_num = 64  # Number of channels
    length_epoch = int(length_epoch_channel / channel_num)
    
    # Reshape your original data into (epoch_num, 64, length_epoch)
    data_3d = Data.reshape(epoch_num, channel_num, length_epoch)  # e.g., 500 * 64 * 20 originally
    data_3d = data_3d.transpose(0, 2, 1)  # Transpose to (epoch_num, length_epoch, channel_num)

    # Original list of channels in your current order
    original_order = [
        "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
        "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
        "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
        "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
        "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
        "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
        "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
        "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2"
    ]

    # Desired layout flattened in the intended order (keeping None for now to handle missing spots)
    desired_layout = [
        None, None, None, "Fp1", "Fpz", "Fp2", None, None, None, 
        None, None, "AF7", "AF3", "AFz", "AF4", "AF8", None, None,
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        None, "P9", "PO7", "PO3", "POz", "PO4", "PO8", "P10", None,
        None, None, "O1", "Oz", "Iz", "O2", None, None, None
        ]
    # Create an empty array for the new layout
    data_reshaped = np.zeros((epoch_num, length_epoch, 9, 9))  # Changed to 9x9 layout
    for i in range(len(desired_layout)):
        row, col = divmod(i, 9)  # Get row and column indices for the 9x9 grid
        channel = desired_layout[i]
        
        if channel and channel.lower() in map(str.lower, original_order):
            original_index = list(map(str.lower, original_order)).index(channel.lower())
            data_reshaped[:, :, row, col] = data_3d[:, :, original_index]
        else:
            if channel:  # If the channel exists but doesn't match
                print(f"Channel {channel} not found in original_order")

    return data_reshaped

def DatasetLoader(DIR):
    # Read Training Data and Labels
    X = pd.read_csv(DIR + 'dataset.csv', header=None)
    X = np.array(X).astype('float32')
    y = pd.read_csv(DIR + 'label.csv', header=None)
    y = np.array(y).astype('float32')
    print(np.shape(X))
    return X, y

class SaliencyMap:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_saliency(self, input_tensor, target):
        input_tensor.requires_grad_()
        output = self.model(input_tensor)
        loss = output[torch.arange(output.shape[0]), target].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        saliency = input_tensor.grad.detach()
        saliency = saliency.squeeze(0).squeeze(0)  

        return saliency.cpu().numpy()  

def shuffle_augmented_data(augmented_X, augmented_y):
    # Convert augmented samples back to NumPy arrays
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    augmented_X, augmented_y = shuffle(augmented_X, augmented_y)
    return augmented_X, augmented_y

def adhoc_augmentation_full_dataset(X_train, y_train, num_trials=10, increase=True, increase_factor=2):
    """
    Performs ad-hoc augmentation by randomly selecting trials, either with or without replacement,
    averaging them, and either adding them to the dataset or replacing the dataset.
    
    Args:
        X_train (numpy array): Original training data of shape (N, C, H, W) or (N, Channels, Height, Width).
        y_train (numpy array): Corresponding labels of shape (N,).
        num_trials (int): Number of trials to average in each new sample.
        increase (bool): If True, add the averaged samples to the original dataset (with replacement).
                         If False, replace the dataset with only the averaged samples (without replacement).
        increase_factor (float): Specifies how much to increase the dataset size as a fraction of the original dataset
                                 size (only used if increase=True).
        
    Returns:
        Augmented X_train and y_train as NumPy arrays with averaged samples either added to or replacing the dataset.
    """
    # Convert to PyTorch tensors if input is in NumPy format
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.long)

    augmented_X = []
    augmented_y = []

    # Get unique labels
    unique_labels = torch.unique(y_train)

    for label in unique_labels:
        # Get indices of all samples with this label
        label_indices = (y_train == label).nonzero(as_tuple=True)[0]
        
        # Check if there are enough samples to average without replacement if required
        if not increase and len(label_indices) < num_trials:
            raise ValueError(f"Not enough samples to select {num_trials} without replacement for label {label}.")
        
        # If increasing the dataset size, use replacement; otherwise, use without replacement
        if increase:
            # Calculate the number of new samples to add based on the increase factor
            num_samples_to_add = int(500 * increase_factor) #int(len(label_indices) * increase_factor)
            # Sampling with replacement
            for _ in range(num_samples_to_add):
                selected_indices = np.random.choice(label_indices, num_trials, replace=True)
                averaged_sample = X_train[selected_indices].mean(dim=0)
                augmented_X.append(averaged_sample.numpy())
                augmented_y.append(label.item())
        else:
            # Sampling without replacement
            shuffled_indices = torch.randperm(len(label_indices))
            # Iterate through the label indices in chunks of `num_trials`
            for i in range(0, len(label_indices) - num_trials + 1, num_trials):
                selected_indices = label_indices[shuffled_indices[i:i + num_trials]]
                averaged_sample = X_train[selected_indices].mean(dim=0)
                augmented_X.append(averaged_sample.numpy())
                augmented_y.append(label.item())

    # Convert augmented samples back to NumPy arrays
    augmented_X,augmented_y =  shuffle_augmented_data(augmented_X, augmented_y)
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    return augmented_X, augmented_y

def train_model(training_mode, X, y, num_epochs, points, fc_features, max_restarts, par, stim, task, save_dir, evaluate, 
                                                    adhoc_augmentation_full_dataset, num_trials, increase_factor_train, batch_size, normalize):

    save_dir_models = f"{save_dir}/models"
    save_dir_sals = f"{save_dir}/sals"
    save_dir_csvs = f"{save_dir}/csvs"
    save_dir_plots = f"{save_dir}/plots"
    
    os.makedirs(save_dir_models,exist_ok=True)
    os.makedirs(save_dir_sals,exist_ok=True)
    os.makedirs(save_dir_csvs,exist_ok=True)
    os.makedirs(save_dir_plots,exist_ok=True)

    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = 40
    len_trial = 0

    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]


    metrics_df = pd.DataFrame()
    fold_metrics = []

    stacked_images_val_0 = []
    stacked_images_val_1 = []
    stacked_images_train_0 = []
    stacked_images_train_1 = []
    ct1,ct2,ct3,ct4 = 0,0,0,0
    
    for iteration in range(max_restarts):
        random_state = np.random.RandomState(seed=base_seed + iteration)
        random.seed(base_seed + iteration)  # Python random
        np.random.seed(base_seed + iteration)  # NumPy
        torch.manual_seed(base_seed + iteration)  # PyTorch CPU
        torch.cuda.manual_seed(base_seed + iteration)  # PyTorch GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"Iteration {iteration + 1}")
        
        val_idx_class_0 = random_state.choice(idx_class_0, num_trials, replace=False)
        val_idx_class_1 = random_state.choice(idx_class_1, num_trials, replace=False)
        val_idx = np.concatenate([val_idx_class_0, val_idx_class_1])

        train_idx_class_0 = np.setdiff1d(idx_class_0, val_idx_class_0)
        train_idx_class_1 = np.setdiff1d(idx_class_1, val_idx_class_1)
        train_idx = np.concatenate([train_idx_class_0, train_idx_class_1])

        X_train_b, X_val_b = X[train_idx], X[val_idx]
        y_train_b, y_val_b = y[train_idx], y[val_idx]

        X_train, y_train = adhoc_augmentation_full_dataset(X_train_b, y_train_b, num_trials, increase=True, increase_factor=increase_factor_train)
        X_val, y_val = adhoc_augmentation_full_dataset(X_val_b, y_val_b, num_trials, increase=False, increase_factor=1)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)  # Ensure labels are long for CrossEntropyLoss
        )

        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Initialize the model
        net = EEGNet(fc_features).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        
        dataset_size = len(train_loader.dataset)
        subset_size = dataset_size // num_epochs  

        if training_mode:
            for epoch in range(num_epochs):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                net.train()
                running_loss = 0.0

                start_idx = epoch * subset_size
                end_idx = dataset_size if epoch == num_epochs - 1 else (epoch + 1) * subset_size
                indices = list(range(start_idx, end_idx))
                subset_train_loader = DataLoader(Subset(train_loader.dataset, indices), batch_size=batch_size, shuffle=True)

                for inputs, labels in subset_train_loader:
                    inputs, labels = inputs.to(device), labels.to(device).long()
                    # **Fix: Remove extra dimension**
                    if inputs.dim() == 6:  # If shape is (batch, 1, 1, time_steps, 9, 9)
                        inputs = inputs.squeeze(2)  # Remove the redundant dim

                    optimizer.zero_grad()
                    outputs = net(inputs)


                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()

                e_train = evaluate(net, subset_train_loader, ["acc", "auc"])
                e_val = evaluate(net, val_loader, ["acc", "auc"])
                scheduler.step(e_val[1])

            new_row = pd.DataFrame({
                'Iteration': [iteration + 1],
                'Epoch': [epoch + 1],
                'Train Size': [len(y_train)],
                'Val Size': [len(y_val)],
                'Train ACC': [e_train[0]], 'Train AUC': [e_train[1]],
                'Val ACC': [e_val[0]], 'Val AUC': [e_val[1]],
            })
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
            last_epoch_val_auc = metrics_df[metrics_df['Epoch'] == num_epochs]['Val AUC'].values[-1]
            last_epoch_val_acc = metrics_df[metrics_df['Epoch'] == num_epochs]['Val ACC'].values[-1]
            last_epoch_train_auc = metrics_df[metrics_df['Epoch'] == num_epochs]['Train AUC'].values[-1]
            last_epoch_train_acc = metrics_df[metrics_df['Epoch'] == num_epochs]['Train ACC'].values[-1]

            fold_metrics.append({
            'Iteration': iteration + 1, 
            'Last Epoch Train AUC': last_epoch_train_auc, 
            'Last Epoch Train ACC': last_epoch_train_acc,
            'Last Epoch Val AUC': last_epoch_val_auc,
            'Last Epoch Val ACC': last_epoch_val_acc
        })
            
        if(training_mode and e_val[0]==1 and e_val[1]==1):
            torch.save(net.state_dict(), os.path.join(save_dir_models, f'{stim}_{task}_best_{par}_{iteration}.pth'))
            
            image_v0, c1  = generate_gradcam_images_per_participant(net, par, iteration, num_trials, points, 0, fc_features, stim, task, save_dir_models, save_dir_sals, X_val, y_val, device, evaluate, GradCAM=SaliencyMap, direction=direction, training_mode = True)
            if np.any(image_v0 >0):
                stacked_images_val_0.append(image_v0)
                ct1 = ct1+c1

            image_v1, c2 = generate_gradcam_images_per_participant(net, par, iteration, num_trials, points, 1, fc_features, stim, task, save_dir_models, save_dir_sals, X_val, y_val, device, evaluate, GradCAM=SaliencyMap, direction=direction, training_mode = True)
            if np.any(image_v1 >0):
                stacked_images_val_1.append(image_v1)
                ct2 = ct2+c2

            image_t0, c3 = generate_gradcam_images_per_participant(net, par,iteration,num_trials,points,0,fc_features,stim,task,save_dir_models, save_dir_sals,X_train,y_train,device,evaluate,GradCAM=SaliencyMap,direction=direction, training_mode = True)
            if np.any(image_t0 >0):
                stacked_images_train_0.append(image_t0)
                ct3 = ct3+c3

            image_t1, c4 = generate_gradcam_images_per_participant(net, par,iteration,num_trials,points,1,fc_features,stim,task,save_dir_models, save_dir_sals,X_train,y_train,device,evaluate,GradCAM=SaliencyMap,direction=direction, training_mode = True)
            if np.any(image_t1 >0):
                stacked_images_train_1.append(image_t1)
                ct4 = ct4+c4
            
    
    fold_metrics_df = pd.DataFrame(fold_metrics)
    df_results = pd.DataFrame(fold_metrics_df, columns=['Iteration', 'Last Epoch Train ACC', 'Last Epoch Train AUC', 'Last Epoch Val ACC', 'Last Epoch Val AUC'])
    df_results.to_csv(os.path.join(save_dir_csvs, f'{par}_results_{stim}_{task}.csv'), index=False)
    
    plotting_gradcam(stacked_images_train_0, 'Training', len_trial, par, points, 0, stim, task, save_dir_sals, save_dir_plots,  direction, ct3, plot_mac=True)
    plotting_gradcam(stacked_images_train_1, 'Training', len_trial, par, points, 1, stim, task, save_dir_sals, save_dir_plots,  direction, ct4, plot_mac=True)
    plotting_gradcam(stacked_images_val_0, 'Validation', len_trial, par, points, 0, stim, task, save_dir_sals, save_dir_plots,  direction, ct1, plot_mac=True)
    plotting_gradcam(stacked_images_val_1, 'Validation', len_trial, par, points, 1, stim, task, save_dir_sals, save_dir_plots,  direction, ct2, plot_mac=True)

    return fold_metrics_df

def generate_gradcam_images_per_participant(net, par, iteration, num_trials, points, target, fc_features, stim, task, save_dir_models, save_dir_sals, X_test, y_test, device, evaluate, GradCAM, direction, training_mode):
    desired_layout = [
        None, None, None, "Fp1", "Fpz", "Fp2", None, None, None, 
        None, None, "AF7", "AF3", "AFz", "AF4", "AF8", None, None,
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        None, "P9", "PO7", "PO3", "POz", "PO4", "PO8", "P10", None,
        None, None, "O1", "Oz", "Iz", "O2", None, None, None
        ]

    layout_mask = np.array([0 if loc is None else 1 for loc in desired_layout]).reshape(9, 9)
    
    if(training_mode):
        best_model = EEGNet(fc_features).to(device)
        best_model.load_state_dict(torch.load(os.path.join(save_dir_models, f'{stim}_{task}_best_{par}_{iteration}.pth'), map_location=device))
    else:
        best_model = net
        
    if(len(X_test)==2*num_trials):
        trial = 'Validation'
        # X_test, y_test = adhoc_augmentation_full_dataset(X_test, y_test, num_trials, increase=False, increase_factor=1)
    else:
        trial = 'Training'
        # X_test, y_test = adhoc_augmentation_full_dataset(X_test, y_test, num_trials, increase=True, increase_factor=1)

    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(device)
    # y_test = y_test.argmax(axis=1)
    
    cams = []
    # Generate Grad-CAM images for correctly predicted samples
    counter = 0
    for i in range(len(X_test)):
        if(y_test[i].item() != target):
            continue
        # print("len_X_test:  ", len(X_test))
        input_image = X_test[i].unsqueeze(0)
        output = best_model(input_image)
        pred = output.argmax(axis=1)

        if (pred.item() == target):
            grad_cam = SaliencyMap(best_model)
            cam = grad_cam.generate_saliency(input_image, target)
            # grad_cam = GradCAM(model=best_model, target_layer='separable_conv_pointwise')  #separable_conv_pointwise #separable_conv_depthwise
            # cam = grad_cam.generate_cam(input_image, target, points = points)
            cam = cam * layout_mask
            # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cam / (np.max(np.abs(cam)) + 1e-8)
            # cam = 2 * (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) - 1

            # input_image = input_image.numpy()
            # input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
            cam = cam  #*input_image #change this if you want to multiply by image
            if np.any(cam > 0):
                cams.append(cam)
                counter+=1
            # else:
            #     print(f"{target} zero_added!")
        else:
            if(trial=='Validation' and (y_test[i].item() != pred.item())):
                print(f"{target} mistake!")
                # print(f"ytest: {y_test[i].item()} and pred: {pred[0][0]} and target:{target}")
                    
    # Convert CAMs to tensors and handle cases with no valid CAMs
    # cams = [torch.from_numpy(cam) if isinstance(cam, np.ndarray) else cam for cam in cams]
    converted_cams = []
    for cam in cams:
        if isinstance(cam, np.ndarray):  # Check if the item is a NumPy array
            converted_cams.append(torch.from_numpy(cam))  # Convert NumPy array to PyTorch tensor
        else:
            converted_cams.append(cam)
    cams = converted_cams

    # if not isinstance(input_image, torch.Tensor):
    #     input_image = torch.tensor(input_image, device=device)
        
    if len(cams) > 0:
        stacked_cams = torch.stack(cams)  # Shape: (num_cams, 1, H, W)
        average_cam = stacked_cams.mean(dim=0)  # Averaging over the stacked dimension, shape: (1, H, W)
        len_trial = stacked_cams.shape[0]
    else:
        # print(f"No valid CAMs generated for participant {par}, target {target}. Returning zero images.")
        average_cam = torch.zeros((points, 9, 9))  # Default zero tensor
        len_trial = 0

    images = average_cam.squeeze().cpu().numpy()  # Shape: [20, 9, 9]
    if(counter != len_trial):
        print("error!")
    return images, counter

def plotting_gradcam(images, trial, len_trial, par, points, target, stim, task, save_dir_sals, save_dir_plots, direction, counter, plot_mac):

    filename = os.path.join(save_dir_sals, f'{stim}_{task}_{direction(task, target)}_{trial}_{par}')
    np.save(f'{filename}.npy', images)

    if(plot_mac):
        desired_layout = [
        None, None, None, "Fp1", "Fpz", "Fp2", None, None, None, 
        None, None, "AF7", "AF3", "AFz", "AF4", "AF8", None, None,
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        None, "P9", "PO7", "PO3", "POz", "PO4", "PO8", "P10", None,
        None, None, "O1", "Oz", "Iz", "O2", None, None, None
        ]

        converted_images = []
        for img in images:
            if isinstance(img, np.ndarray):  # Check if the item is a NumPy array
                converted_images.append(torch.from_numpy(img))  # Convert NumPy array to PyTorch tensor
            else:
                converted_images.append(img)
        images = converted_images

        images = torch.stack(images)  # Shape: (num_cams, 1, H, W)
        images = images.mean(dim=0)  # Averaging over the stacked dimension, shape: (1, H, W)
        
        mask = torch.ones(images.shape)  
        for row in range(9):
            for col in range(9):
                index = row * 9 + col
                if desired_layout[index] is None:
                    # Apply zero mask at the corresponding region
                    mask[:, row, col] = 0
        images = images * mask 

        fig, axes = plt.subplots(5, 5, figsize=(15, 15))

        axes = axes.flatten()

        # Calculate the global min and max values for all images for proper colormap scaling
        global_min = images.min()
        global_max = images.max()
        
        lenn = points
        if stim == 'saccade':
            labels = np.arange(-lenn, 0) * 4 - 4
        else:
            labels = np.arange(0, lenn, 1) * 4 
        
        plt.title(f'{stim}_{task}_{trial}, num_grads:{counter}, direction:{direction(task, target)}')
        # Set a global title
        plt.suptitle(f'{stim}_{task}_{trial}, num_grads:{counter}, direction:{direction(task, target)}', fontsize=16)

        for i in range(images.shape[0]):
            ax = axes[i]
            ax.imshow(images[i], cmap='viridis', vmin=global_min, vmax=global_max)

            # Superimpose text from the layout onto the grid
            for row in range(9):
                for col in range(9):
                    text = desired_layout[row * 9 + col]
                    if text is not None:
                        ax.text(col, row, text, ha='center', va='center', color='white', fontsize=8, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{labels[i]}ms")

        filename = os.path.join(save_dir_plots, f'{stim}_{task}_{direction(task, target)}_{trial}_{par}_all_{counter}.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    return filename
