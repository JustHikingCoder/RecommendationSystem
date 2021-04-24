import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_model
import csv

G_losses = []
D_losses = []

with open('result/G_losses.csv','r') as f:
    reader = csv.reader(f)
    G_losses = [row[1] for row in reader]

with open('result/D_losses.csv','r') as f:
    reader = csv.reader(f)
    D_losses = [row[1] for row in reader]

del(G_losses[0])
del(D_losses[0])

G_losses = [float(i) for i in G_losses]
D_losses = [float(i) for i in D_losses]

plt.plot(range(pytorch_model.num_epochs), G_losses, marker='.', label='G_loss')
plt.plot(range(pytorch_model.num_epochs), D_losses, marker='o', label='D_loss')
plt.title('LOSS')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.show()
