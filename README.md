# Long-Range Transcranial Magnetic Stimulation (LR-TMS)


<p align="center">
  
<img src="Long-Range TMS.png" alt="Long-Range TMS" width="280"/>

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

---
---


# PARTE 2: Sistema para la optimización de diseño electromagnético basado en aprendizaje automático y simulación. 

## Algunas ideas para abordarlo incluyen:
Problema de **optimización de diseño electromagnético** basado en **aprendizaje automático y simulación**.


---

### **1. Aprendizaje Evolutivo + Redes Generativas (GANs)**
📌 **Concepto:**  
Utilizar **algoritmos evolutivos** o **redes generativas adversarias (GANs)** para explorar múltiples configuraciones de bobinas y circuitos.  
📌 **Cómo funciona:**  
- Se generan miles de diseños de bobinas y circuitos en un entorno **paramétrico 3D** (como en OpenSCAD o Blender).  
- Se simulan sus propiedades magnéticas con **Método de Elementos Finitos (FEM)** o **Maxwell Equations Solver**.  
- Se optimizan las configuraciones con **Genetic Algorithms (GA)** o **Bayesian Optimization**.  
- Se usa una GAN para aprender patrones de diseño eficientes.  

📌 **Herramientas recomendadas:**  
- **OpenSCAD + Python API** → para generar modelos 3D.  
- **Elmer FEM** o **COMSOL** → para simulaciones electromagnéticas.  
- **PyTorch/TensorFlow + GANs** → para aprendizaje de formas óptimas.  
- **DEAP (Distributed Evolutionary Algorithms in Python)** → para optimización evolutiva.  

---

### **2. Modelado Diferenciable con Deep Learning**  
📌 **Concepto:**  
Entrenar una **red neuronal diferencial** que optimice el diseño en base a derivadas de campo magnético.  
📌 **Cómo funciona:**  
- Se define una arquitectura de bobina en un entorno **diferenciable** (como DiffCAD).  
- Se entrena una red neuronal para **maximizar la intensidad de campo a 2 metros**.  
- Se aplican técnicas de **Neural Implicit Representations (NeRF)** para generar geometrías optimizadas.  

📌 **Herramientas recomendadas:**  
- **JAX (Google DeepMind)** → para cálculos diferenciables.  
- **Neural Fields (SDF-based 3D representations)** → para diseño paramétrico optimizado.  
- **Blender Python API + PyTorch3D** → para visualización avanzada.  

---

### **3. Reinforcement Learning (RL) para Optimización de Diseño**  
📌 **Concepto:**  
Formular el problema como un **juego de exploración**, donde un agente **intenta mejorar el diseño** en cada iteración.  
📌 **Cómo funciona:**  
- Se usa un **agente RL** que prueba **diferentes geometrías y configuraciones electrónicas**.  
- Cada diseño es simulado y recibe una **recompensa** basada en la intensidad del campo magnético en la distancia objetivo.  
- Se usan técnicas como **Deep Q-Networks (DQN)** o **Proximal Policy Optimization (PPO)** para mejorar el diseño en cada iteración.  

📌 **Herramientas recomendadas:**  
- **Stable-Baselines3 (SB3)** → para entrenar agentes RL.  
- **PyBullet / Mujoco** → para simulaciones de campo electromagnético en entorno físico.  
- **Blender + RL API** → para modelado dinámico.  

---

