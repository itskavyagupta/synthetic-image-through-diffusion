import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import ViTFeatureExtractor
from tensorflow.keras.preprocessing import image as keras_image

from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score


def main(args_user):
    """## Loading the data"""

    # Initialize the ViT feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Create a custom transformation pipeline
    transform = transforms.Compose([
        transforms .Resize((224, 224)),  # Resize images to 224x224
        transforms .ToTensor(),
        lambda x: x * 255,  # Scale pixel values
        transforms .Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)  # Normalize using predefined mean and std
    ])


    full_dataset = datasets.ImageFolder(root=args_user.root_baseline, transform=transform)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    def collate_fn(batch):
        pixel_values = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}

    id2label = {0: 'benign', 1: 'malignant'}
    label2id = {v: k for k, v in id2label.items()}

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        id2label=id2label,
        label2id=label2id
    )


    # Define the training arguments
    args = TrainingArguments(
        output_dir="vit-classification-task",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='logs',
        save_strategy="epoch",
        report_to="none"
    )

    # Define a simple compute_metrics function


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {'accuracy': accuracy}

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Run training
    trainer.train()

    # Retrieve and plot accuracy logs
    accuracy_logs = [log['eval_accuracy'] for log in trainer.state.log_history if 'eval_accuracy' in log]
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_logs, marker='o', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('accuracy_over_epochs.png')

    """Hybrid dataset"""

    # Initialize the ViT feature extractor
    feature_extractor_hybrid = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Create a custom transformation pipeline
    transform = transforms .Compose([
        transforms .Resize((224, 224)),  # Resize images to 224x224
        transforms .ToTensor(),
        lambda x: x * 255,  # Scale pixel values
        transforms .Normalize(mean=feature_extractor_hybrid.image_mean, std=feature_extractor_hybrid.image_std)  # Normalize using predefined mean and std
    ])

    full_dataset_hybrid = datasets.ImageFolder(root=args_user.root_hybrid, transform=transform)

    # Split the dataset into train and test sets
    train_size_hybrid = int(0.8 * len(full_dataset_hybrid))
    test_size_hybrid = len(full_dataset_hybrid) - train_size_hybrid
    train_dataset_hybrid, test_dataset_hybrid = random_split(full_dataset_hybrid, [train_size_hybrid, test_size_hybrid])

    model_hybrid = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        id2label=id2label,
        label2id=label2id
    )

    # Initialize the Trainer
    trainer_hybrid = Trainer(
        model=model_hybrid,
        args=args,
        train_dataset=train_dataset_hybrid,
        eval_dataset=test_dataset_hybrid,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Run training
    trainer_hybrid.train()

    # Retrieve and plot accuracy logs
    accuracy_logs_hybrid = [log['eval_accuracy'] for log in trainer_hybrid.state.log_history if 'eval_accuracy' in log]
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_logs_hybrid, marker='o', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('accuracy_over_epochs_hybrid.png')

    """Test dataset"""

    test_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Define the transformation for test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
        transforms.ToTensor(),  # Convert the images to PyTorch tensors and scale to [0, 1]
        transforms.Normalize(mean=test_feature_extractor.image_mean, std=test_feature_extractor.image_std)  # Normalize the images
    ])

    # Load the test dataset with unique names
    test_data_folder = args_user.root_test
    test_image_dataset = datasets.ImageFolder(root=test_data_folder, transform=test_transform)

    # Create a DataLoader for the test dataset with a unique name
    test_image_loader = DataLoader(test_image_dataset, batch_size=32, shuffle=False)

    def evaluate_model_on_test(test_model, test_data_loader):
        test_model.eval()  # Ensure the model is in evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_model.to(device)

        test_predictions = []
        test_true_labels = []

        with torch.no_grad():  # Disable gradient computation during inference
            for test_batch_images, test_batch_labels in test_data_loader:
                test_batch_images = test_batch_images.to(device)
                test_batch_labels = test_batch_labels.to(device)
                test_outputs = test_model(test_batch_images)
                test_batch_preds = torch.argmax(test_outputs.logits, dim=1)

                test_predictions.extend(test_batch_preds.cpu().numpy())
                test_true_labels.extend(test_batch_labels.cpu().numpy())

        return test_predictions, test_true_labels

    test_preds, test_labels = evaluate_model_on_test(model, test_image_loader)

    # Calculate and print the accuracy of the model on the test data

    test_accuracy = accuracy_score(test_labels, test_preds)
    print(f"Test Accuracy of baseline dataset trained ViT: {test_accuracy:.4f}")

    test_preds_hybrid, test_labels_hybrid = evaluate_model_on_test(model_hybrid, test_image_loader)

    # Calculate and print the accuracy of the model on the test data

    test_accuracy_hybrid = accuracy_score(test_labels_hybrid, test_preds_hybrid)
    print(f"Test Accuracy of hybrid dataset trained ViT: {test_accuracy_hybrid:.4f}")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Load an image dataset")
    parser.add_argument('--root_baseline', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--root_hybrid', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--root_test', type=str, required=True, help='Root directory for the dataset')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
