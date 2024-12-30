import os
import json
import torch
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch.nn as nn
from typing import List
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Loss functions
        self.loss_function = F.binary_cross_entropy
        self.loss_function_params = {'reduction': 'sum'}
        # Loss tracking variables
        self.train_loss_array: List[float] = []
        self.validation_loss_array: List[float] = []
        self.train_epoch_loss = 0
        self.validation_epoch_loss = 0
        self.train_batch_idx = 0
        self.validation_batch_idx = 0
        # Metrics
        self.metrics = nn.ModuleDict()
        self.train_time_array: List[float] = []
        self.validation_time_array: List[float] = []
        self._timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    def get_timestamp(self):
        return self._timestamp

    def setup_metrics(self, threshold=0.5):
        """
        Initialize the metrics for evaluation.
        """
        self.metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold)
        })

    def reset_metrics(self):
        """
        Reset all metrics before evaluation.
        """
        for key, metric in self.metrics.items():
            metric.reset()

    def update_metrics(self, predictions, targets):
        """
        Update metrics with predictions and targets.

        Args:
            predictions (Tensor): Model predictions.
            targets (Tensor): Ground truth labels.
        """
        for metric in self.metrics.values():
            metric.update(predictions.flatten(), targets.flatten().int())

    def compute_metrics(self):
        """
        Compute and return the metrics as a dictionary.

        Returns:
            dict: Dictionary of computed metrics.
        """
        return {key: metric.compute().item() for key, metric in self.metrics.items()}

    def evaluate(self, loader, device, multimodal=False):
        """
        Evaluate the model on the given data loader.

        Args:
            loader (DataLoader): DataLoader for evaluation.
            device (torch.device): Device to run the evaluation.

        Returns:
            dict: Computed metrics as a dictionary.
        """
        self.eval()
        self.reset_metrics()
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
                x = tuple(module.to(device) for module in x)
                y_pred = self(x)
                target = y.to(device)
                self.update_metrics(y_pred, target)
        return self.compute_metrics()

    def train_model(self, train_loader, optimizer, device, multimodal=False):
        """
        Train the model on the given data loader.

        Args:
            loader (DataLoader): DataLoader for training.
            optimizer (Optimizer): Optimizer for training.
            device (torch.device): Device to run the training.
        """
        self.train()
        for x, y in tqdm(train_loader, desc="Training".ljust(10), leave=False, dynamic_ncols=True):
            self.zero_grad()
            x = tuple(module.to(device) for module in x)
            y_pred = self(x)
            loss = self.train_loss(y_pred, y.to(device))
            loss.backward()
            optimizer.step()

    def validate_model(self, validation_loader, device, multimodal=False):
        """
        Validate the model on the given data loader.
        Args:
            loader (DataLoader): DataLoader for validation.
            device (torch.device): Device to run the validation.
        """
        self.eval()
        with torch.no_grad():
            for x, y in tqdm(validation_loader, desc="Validation".ljust(10), leave=False, dynamic_ncols=True):
                x = tuple(module.to(device) for module in x)
                y_pred = self(x)
                self.validation_loss(y_pred, y.to(device))

    def train_loss(self, predictions, targets):
        try:
            loss = self.loss_function(
                    input=predictions,
                    target=targets,
                    **self.loss_function_params
            )
            self.train_epoch_loss += loss.item()
            self.train_batch_idx += 1
            return loss
        except Exception as e:
            print(f'Prediction shape: {predictions.shape}')
            print(f'Target shape: {targets.shape}')
            raise e

    def validation_loss(self, predictions, targets):
        loss = self.loss_function(
                input=predictions,
                target=targets,
                **self.loss_function_params
        )

        self.validation_epoch_loss += loss.item()
        self.validation_batch_idx += 1
        return loss

    def reset_loss(self):
        # Append averaged losses and reset accumulators
        if self.train_batch_idx > 0:
            self.train_loss_array.append(self.train_epoch_loss / self.train_batch_idx)
        self.train_epoch_loss = 0
        self.train_batch_idx = 0

        if self.validation_batch_idx > 0:
            self.validation_loss_array.append(self.validation_epoch_loss / self.validation_batch_idx)
        self.validation_epoch_loss = 0
        self.validation_batch_idx = 0

    def save_loss_plot(self, save_dir, epochs):
        # Extract class/module name for identification
        module_path = self.__class__.__module__
        model_identifier = module_path.split('.')[-1]

        # Directory path with date
        data_dir = os.path.join(save_dir, model_identifier, self.get_timestamp())
        os.makedirs(data_dir, exist_ok=True)

        # File name and path
        file_name = f"{model_identifier}_{self._timestamp}_loss_plot.png"
        file_path = os.path.join(data_dir, file_name)

        # Plot and save the losses
        plt.plot(self.train_loss_array, label='Train Loss')
        if self.validation_loss_array:
            plt.plot(self.validation_loss_array, label='Validation Loss')
            plt.title('Train and Validation Loss')
        else:
            plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(file_path)
        plt.close()
        print(f"Loss plot saved to: {file_path}")

    def save_results(self, results, save_dir, epochs):
        """
        Save evaluation results to a JSON file.

        Args:
            results (dict): Dictionary containing evaluation results.
            save_dir (str): Directory to save the results file.
            model_name (str): Name of the model for the file naming.
        """
        # Extract class/module name for identification
        module_path = self.__class__.__module__
        model_identifier = module_path.split('.')[-1]

        # Directory path with date
        data_dir = os.path.join(save_dir, model_identifier, self.get_timestamp())
        os.makedirs(data_dir, exist_ok=True)

        # File name and path
        file_name = f"{model_identifier}_{self._timestamp}_results.json"
        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "w") as f:
            json.dump(results, f)
        print(f"Evaluation results saved to: {file_path}")

    def save_weights(self, save_dir, epochs):
        """
        Save model weights to a file.
        Args:
            save_dir (str): Directory to save the model weights.
        """
        # Extract class/module name for identification
        module_path = self.__class__.__module__
        model_identifier = module_path.split('.')[-1]
        # Directory path with date
        data_dir = os.path.join(save_dir, model_identifier, self.get_timestamp())
        os.makedirs(data_dir, exist_ok=True)
        # File name and path
        file_name = f"{model_identifier}_{self._timestamp}_weights.pth"
        file_path = os.path.join(data_dir, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model weights saved to: {file_path}")

