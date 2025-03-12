# Long-Range Transcranial Magnetic Stimulation (TMS)


<p align="center">
  
<img src="Long-Range TMS.png" alt="Long-Range TMS" width="400"/>

</p>

## Overview
This project explores the development of a **Long-Range Transcranial Magnetic Stimulation (TMS) system**. Unlike conventional TMS devices that function effectively within 2 to 5 cm from the scalp, this system aims to operate beyond 2 meters.. The system leverages **neural networks** to analyze and predict the optimal **electromagnetic configurations** required for long-range operation.

## Features
- **Deep learning-based simulation** of electromagnetic fields.
- **Optimization of coil configurations** for long-range stimulation.
- **Data-driven predictions** of required electrical parameters.
- **Python-based framework** utilizing TensorFlow/PyTorch.

## Installation
```bash
pip install numpy scipy torch matplotlib
```

## Usage

```python
python train_model.py
```

---

# Python code: **train_model.py**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Simulated dataset (input: coil parameters, output: effective magnetic field at distance)
def generate_data(num_samples=1000):
    np.random.seed(42)
    coil_turns = np.random.uniform(100, 1000, num_samples)
    current = np.random.uniform(0.1, 10, num_samples)
    frequency = np.random.uniform(10, 1000, num_samples)
    distance = np.random.uniform(2, 3, num_samples)  # Ensuring long-range scenarios
    
    # Magnetic field approximation (dummy function, replace with physics-based model)
    B_field = (coil_turns * current * frequency) / (distance**2 + 1)
    
    X = np.stack([coil_turns, current, frequency, distance], axis=1)
    y = B_field.reshape(-1, 1)
    
    return X, y

class TMSPredictor(nn.Module):
    def __init__(self):
        super(TMSPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Generate dataset
X, y = generate_data()
X_train, y_train = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define model and optimizer
model = TMSPredictor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
def train(epochs=500):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

train()

# Save model
torch.save(model.state_dict(), "tms_model.pth")
print("Model trained and saved successfully!")

```

Good luck !! 🚀
