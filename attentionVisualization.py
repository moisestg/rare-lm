import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import numpy as np

data = np.expand_dims(
	np.array([
	2.44148232e-05,6.74716949e-01,9.05486161e-07
	,1.96977780e-07,3.14657655e-07,7.50977662e-08,6.44223732e-08
	,3.82127155e-07,2.82049632e-06,3.70216355e-07,8.86810176e-06])
, axis=0)

data *= 1/(1-0.279215)

cmap = mpl.cm.seismic
norm = mpl.colors.Normalize(vmin=0., vmax=1.)

fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

ax.set_xticks(np.arange(data.shape[1]));

ax.set_xticklabels(
	["the","fort","?","what","do","you","mean",",","out","of","the"]
, rotation=45, ha='right')

ax.set_yticks([])

plt.show()