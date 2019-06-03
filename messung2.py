import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def csv_read(pathToFile, delimiter=";"):
    with open(pathToFile, "r") as f:
        content = []
        for line in f:
            content.append((line.rstrip()).split(delimiter))
    return content

werte = csv_read("csv/messung2.csv")
xdata = np.zeros(21)
ydata = np.zeros(21)
ignore = True
i=0

for values in werte:
    if(ignore):
        ignore = False
    else:
        xdata[i] = float(values[0])
        ydata[i] = float(values[1])
        i+=1

x_line = np.linspace(0, 1080)
plt.plot(xdata, ydata,  "rx", label="Messwerte")

plt.grid()
plt.xlabel(r"$U$ / V")
plt.ylabel(r"$I$ / A")


plt.legend()
plt.tight_layout()
plt.savefig("build/messung2.pdf")