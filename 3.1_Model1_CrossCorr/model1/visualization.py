import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

point_indices = [[17, 15, 0, 1, 8, 9, 10, 11, 22, 23] , [4, 3, 2, 1, 5, 6, 7], [8, 12, 13, 14, 19, 20], [0, 16, 18], [11, 24], [14, 21]]

y_prevs = []

def update_animation(i, point_indices, data, lines, threshold):
	for (indices, line) in zip(point_indices, lines):
		print(data[i, indices, 0], data[i, indices, 1])
		line.set_data(data[i, indices, 0], data[i, indices, 1])
	return lines

def show_sequence(data, fps = 25, output_path = None, threshold = 0.2):
	fig, ax = plt.subplots()
	lines = [ax.plot([], [])[0] for ___ in point_indices]
	d1 = data.reshape(-1, 3)
	d1i = np.where(d1[ : , 2] >= threshold)
	ax.set(xlim = (np.min(d1[d1i, 0]), np.max(d1[d1i, 0])), xlabel = "X")
	ax.set(ylim = (np.min(d1[d1i, 1]), np.max(d1[d1i, 1])), ylabel = "Y")

	ani = animation.FuncAnimation(fig, update_animation,
		fargs = (point_indices, data, lines, threshold), frames = np.arange(len(data)), interval = 1000 / fps, repeat = False)

	if output_path is None:
		plt.show()
	else:
		writer = animation.writers["ffmpeg"](fps = fps)
		ani.save(output_path, writer = writer)
