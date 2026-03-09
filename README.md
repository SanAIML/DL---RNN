# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
To develop a Recurrent Neural Network (RNN) model to predict stock prices based on historical data. This involves loading and preprocessing stock data, defining and training an RNN model, and then evaluating its performance by comparing predicted prices with actual prices.


## DESIGN STEPS
### STEP 1: 

Data Loading and Preprocessing: Load the training and test datasets, extract the 'Close' prices, normalize them using MinMaxScaler, create sequences for the RNN, and convert the data into PyTorch tensors and DataLoaders for training.

### STEP 2: 
Define RNN Model Architecture: Define the RNNModel class, specifying the layers (RNN and Linear) and their connections. Instantiate the model and move it to the appropriate device (CPU or GPU).


### STEP 3: 
Model Compilation: Set up the loss function (e.g., Mean Squared Error) and the optimizer (e.g., Adam) that will be used during the model training phase.


### STEP 4: 
Train the RNN Model: Implement the training loop to iterate over epochs, process data in batches, compute predictions, calculate loss, perform backpropagation, and update model weights. Track and visualize the training loss over epochs.


### STEP 5: 

Make Predictions and Evaluate: Use the trained model to generate predictions on the test dataset. Inverse transform the scaled predictions and actual values to their original price scale. Visualize the actual vs. predicted stock prices and print the final predicted and actual values.

## PROGRAM

### Name: Sanchita Sandeep

### Register Number: 212224240142

```python
class RNNModel(nn.Module):
    # write your code here
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
     super(RNNModel,self).__init__()
     self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
     self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
     out,_=self.rnn(x)
     out=self.fc(out[:,-1,:])
     return out

def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch , y_batch in train_loader:
      x_batch , y_batch = x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs , y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f'Epoch[{epoch+1}/{epochs}],Loss:{total_loss/len(train_loader):.4f}')
  print('Name: Sanchita Sandeep                ')
  print('Register Number: 212224240142    ')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
  
train_model(model, train_loader, criterion, optimizer)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="397" height="301" alt="Screenshot 2026-03-09 092846" src="https://github.com/user-attachments/assets/a29ef3f2-033f-45f3-8757-6d21a2ec9df1" />


## True Stock Price, Predicted Stock Price vs time

<img width="428" height="299" alt="Screenshot 2026-03-09 092903" src="https://github.com/user-attachments/assets/8eb9aa50-fdf2-4af4-8f9d-a90ad6760055" />


### Predictions


<img width="311" height="40" alt="Screenshot 2026-03-09 092911" src="https://github.com/user-attachments/assets/ff07cf7a-6ace-42a6-a48e-1fc465ec0226" />


## RESULT
The model is created succesfully
