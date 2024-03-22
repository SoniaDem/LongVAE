"""

This file opens the file plotLoss.txt.
loss_file_normal.txt is formatted in the form:
    train: 0.3000\n
    val: 0.3400\n

Command line:
    python plot_loss.py loss_file_normal.txt
"""




import sys
import matplotlib.pyplot as plt

loss_file = sys.argv[1]

lines = [l.strip('\n') for l in open(loss_file, 'r').readlines()]
lines = [l for l in lines if l != '']

train = [float(l.split(' ')[-1]) for l in lines if 'train' in l]
val = [float(l.split(' ')[-1]) for l in lines if 'val' in l]

plt.figure()
ax = plt.subplot(111)
ax.plot(train, color='tab:blue', label='Train')
ax.plot(val, color='tab:orange', label='Val')
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss Throughout Training')
plt.show()

