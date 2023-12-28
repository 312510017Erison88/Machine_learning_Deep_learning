
import json
import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMultipleChoice, BertTokenizerFast, BertTokenizer
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load Model and Tokenizer
#model = BertForMultipleChoice.from_pretrained("best_model").to(device)

model = BertForMultipleChoice.from_pretrained("bert-base-chinese").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# Add Dropout layer and weight decay to the model
dropout_prob = 0.5  # You can adjust this value based on your needs
model.classifier.dropout = torch.nn.Dropout(p=dropout_prob)


maxlength=240
batch_size = 2
num_epochs = 40
learning_rate = 7*1e-6
scheduled_sampling_prob = 0.5  # probability of using ground truth labels

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=maxlength):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["問題"]
        paragraph = item["文章"]
        choices = [str(item[f"選項{i}"]) for i in range(1, 5)]
        correct_answer = item['正確答案']

        prompt = [question + paragraph]
        choice_list = choices
        label = int(correct_answer) - 1

        encodings = self.tokenizer(
            [prompt[0] + choice for choice in choice_list],
            choice_list,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt', # Add this line to get PyTorch tensors directly
        )


        input_ids = encodings['input_ids'].clone().detach().to(device)
        attention_mask = encodings['attention_mask'].clone().detach().to(device)
        token_type_ids = encodings['token_type_ids'].clone().detach().to(device)

        label = torch.tensor(label).to(device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label
        }

# Load data from train.json
with open("train.json", "r", encoding="utf-8") as train_file:
    train_data = json.load(train_file)

# Split data into training and validation sets
train_set, valid_set = train_test_split(train_data, test_size=0.05, random_state=42, shuffle=True)

# Tokenize the datasets
train_dataset = MyDataset(train_set, tokenizer)
valid_dataset = MyDataset(valid_set, tokenizer)

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

from tqdm.auto import tqdm
import torch.nn.functional as F

best_validation_loss = float('inf')  # Initialize with a high value
train_losses = []
validation_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Training loop
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        
        use_ground_truth = random.random() < scheduled_sampling_prob

        # Forward pass
        if use_ground_truth:
            # Use ground truth labels
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
        else:
            # Use model's own predictions
            outputs = model(**inputs)
            logits = outputs.logits
        
        #outputs = model(**inputs, labels=labels)
        #logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted_labels = torch.max(logits, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

    # Calculate and print training loss
    average_train_loss = total_loss / len(train_loader)
    train_losses.append(average_train_loss)
    train_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}, Training Loss: {average_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()
    total_validation_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            outputs = model(**inputs, labels=labels)
            logits = outputs.logits

            # Apply softmax to logits
            probs = F.softmax(logits, dim=-1)
            loss = criterion(probs, labels)

            total_validation_loss += loss.item()

            # Calculate accuracy
            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate and print validation loss
    average_validation_loss = total_validation_loss / len(valid_loader)
    validation_losses.append(average_validation_loss)
    validation_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}, Validation Loss: {average_validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}")

    # Check if this epoch has the best validation loss
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        # Save the best model
        model.save_pretrained("best_model")

print("Training finished.")

import matplotlib.pyplot as plt
# Plotting the training and validation loss curves
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

class MyPredictDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=maxlength):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["問題"]
        paragraph = item["文章"]
        choices = [str(item[f"選項{i}"]) for i in range(1, 5)]

        prompt = [question + paragraph]
        choice_list = choices

        encodings = self.tokenizer(
            [prompt[0] + choice for choice in choice_list],
            choice_list,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        token_type_ids = torch.tensor(encodings['token_type_ids'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }

import json
import pandas as pd
from tqdm.auto import tqdm

# Load the test data
with open("test.json", "r", encoding="utf-8") as test_file:
    test_data = json.load(test_file)

# Create a DataLoader for prediction set
predict_dataset = MyPredictDataset(test_data, tokenizer)
predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Set the model to evaluation mode
model.eval()

# Inference loop for the test data
predictions_list = []

for batch in tqdm(predict_loader, desc="Inference"):
    inputs = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted answer (index with the highest logit)
    predicted_answer = torch.argmax(logits, dim=1).cpu().numpy() + 1

    # Append to the list
    predictions_list.extend(predicted_answer)

# Create DataFrame from the predictions list
predictions_df = pd.DataFrame({"ID": [item["題號"] for item in test_data], "Answer": predictions_list})

# Save predictions to a CSV file
predictions_df.to_csv("predictions.csv", index=False)

print("Prediction finished. Results saved in predictions.csv.")