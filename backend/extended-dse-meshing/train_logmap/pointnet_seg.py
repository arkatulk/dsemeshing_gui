import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Utility functions (replacing `tf_util` as needed)
def safe_norm(x, epsilon=1e-8, axis=None):
    """
    Computes the safe norm of a tensor.

    Args:
        x: Input tensor.
        epsilon: Small value to avoid division by zero.
        axis: Axis along which to compute the norm. If None, computes the norm of the flattened tensor.

    Returns:
        The safe norm of the tensor.
    """
    # Square the elements
    squared_sum = torch.sum(x ** 2, dim=axis)
    # Apply max to avoid sqrt of very small values
    max_val = torch.clamp(squared_sum, min=epsilon)
    # Compute the square root
    return torch.sqrt(max_val)


class classification_net(nn.Module):
    def __init__(self, batch_size=8, activation=F.relu):
        super(classification_net, self).__init__()
        self.batch_size = batch_size
        self.activation = activation

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 4), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))


        self.conv6 = nn.Conv2d(1028, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.conv7 = nn.Conv2d(1024, 528, kernel_size=(1, 1), stride=(1, 1))
        self.conv8 = nn.Conv2d(528, 3, kernel_size=(1, 1), stride=(1, 1))
        self.conv9 = nn.Conv2d(3 + 1 + 1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.conv10 = nn.Conv2d(1024, 528, kernel_size=(1, 1), stride=(1, 1))
        self.conv11 = nn.Conv2d(528, 1, kernel_size=(1, 1), stride=(1, 1))

        # Define fully connected layers
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, point_cloud): # can add is_training variable to forward function if neccessary
        num_point = point_cloud.shape[1]

        # Compute Euclidean distances
        euc_dists = safe_norm(point_cloud - point_cloud[:, 0:1, :], axis=-1).unsqueeze(-1)
        point_cloud = torch.cat([point_cloud, euc_dists], dim=2)
        input_image = point_cloud.unsqueeze(1)  # Add channel dimension 
       
        # Convolutions
        net = self.activation(self.conv1(input_image))
        net = self.activation(self.conv2(net))
        net = self.activation(self.conv3(net))
        net = self.activation(self.conv4(net))
        points_feat1 = self.activation(self.conv5(net))
       
        # Max pooling
        #pc_feat1 = torch.max(points_feat1, dim=1, keepdim=False).values
        pc_feat1=F.max_pool2d(points_feat1, kernel_size=(num_point,1), stride=(1, 1))
     
        # Fully connected layers
        pc_feat1 = pc_feat1.view(self.batch_size, 1024)
        pc_feat1 = self.activation(self.fc1(pc_feat1))
        pc_feat1 = self.activation(self.fc2(pc_feat1))
  
        # Concatenate features
        pc_feat1_expand = pc_feat1.view(self.batch_size,-1, 1, 1).repeat(1,1, num_point, 1)

        points_feat1_concat = torch.cat([point_cloud.unsqueeze(-2).permute(0,3,1,2), pc_feat1_expand], dim=1)
        
        # More convolutions
        net = self.activation(self.conv6(points_feat1_concat))
        net = self.activation(self.conv7(net))
        net = self.conv8(net)
        # Compute additional distances
        euc_dists = safe_norm(net - net[:,:, 0:1, :], axis=1).unsqueeze(1)
        net = torch.cat([net, euc_dists], dim=1)
        # Final convolutions
        points_feat2_concat = torch.cat([net, pc_feat1_expand], dim=1)
        net = self.activation(self.conv9(points_feat2_concat))
        net = self.activation(self.conv10(net))
        net = self.conv11(net)

        net = net.squeeze(2)  # Remove the second-to-last dimension
        return net

class logmap_net(nn.Module):
    def __init__(self, batch_size=8, activation=F.relu, is_training=True):
        super(logmap_net, self).__init__()
        self.is_training = is_training
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 4), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.conv6 = nn.Conv2d(1024 + 4, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.conv7 = nn.Conv2d(1024, 528, kernel_size=(1, 1), stride=(1, 1))
        self.conv8 = nn.Conv2d(528, 3, kernel_size=(1, 1), stride=(1, 1))

        self.conv9 = nn.Conv2d(3 + 1+1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        self.conv10 = nn.Conv2d(1024, 528, kernel_size=(1, 1), stride=(1, 1))
        self.conv11 = nn.Conv2d(528, 2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, point_cloud):
        num_point = point_cloud.shape[1]

        # Compute Euclidean distances
        euc_dists = safe_norm(point_cloud - point_cloud[:, 0:1, :], axis=-1).unsqueeze(-1)
        point_cloud = torch.cat([point_cloud, euc_dists], dim=2)
        input_image = point_cloud.unsqueeze(1)  # Add channel dimension 
       

        net = F.relu(self.conv1(input_image))
        net = F.relu(self.conv2(net))
        net = F.relu(self.conv3(net))
        net = F.relu(self.conv4(net))
        points_feat1 = F.relu(self.conv5(net))
        pc_feat1 = F.max_pool2d(points_feat1, (num_point,1))
        pc_feat1 = pc_feat1.view(self.batch_size, 1024)
        pc_feat1 = F.relu(self.fc1(pc_feat1))
        pc_feat1 = F.relu(self.fc2(pc_feat1))

        pc_feat1_expand = pc_feat1.view(self.batch_size,-1, 1, 1).repeat(1,1, num_point, 1)

        points_feat1_concat = torch.cat([point_cloud.unsqueeze(-2).permute(0,3,1,2), pc_feat1_expand], dim=1)

        net = F.relu(self.conv6(points_feat1_concat))
        net = F.relu(self.conv7(net))
        net = F.relu(self.conv8(net))


        euc_dists = safe_norm(net - net[:,:, 0:1, :], axis=1).unsqueeze(1)
        net = torch.cat([net, euc_dists], dim=1)

        points_feat2_concat = torch.cat([net, pc_feat1_expand], dim=1)
        net = F.relu(self.conv9(points_feat2_concat))
        net = F.relu(self.conv10(net))
        net = self.conv11(net)
        
        net = net.squeeze(2)
        
        return net

