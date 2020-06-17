
import matplotlib.pyplot as plt

file = open('rl_training.csv')

trains = []
vals = []

for row in file:
	if not "Train reward:" in row or not "Val reward:" in row:
		continue
	
	train, val = [float(row.split(' ')[i]) for i in (4, 8)]
	trains.append(train)
	vals.append(val)


fig, ax = plt.subplots()
ax.plot(trains, label='Training')
ax.plot(vals, label='Validation')
ax.legend()
plt.show()

