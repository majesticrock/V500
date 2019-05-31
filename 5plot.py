import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp


def csv_read(pathToFile, delimiter=";"):
    with open(pathToFile, "r") as f:
        content = []
        for line in f:
            content.append((line.rstrip()).split(delimiter))
    return content

def func(x, a, b):
    return a*x + b

farbe = ["blaugruen", "gelb","gruen","rot","violett1","violett2"]

n=np.array([11,11,11,11,11,11])
#Hier Werte der Beschleunigungsspannung
wlaenge =  np.array([491.6, 577.0, 546, 614.95 , 435.8, 404.7])

a = np.zeros(6)
a_err = np.zeros(6)
#ag = unp.zeros(6)
b = np.zeros(6)
b_err = np.zeros(6)
#bg = np.zeros(6)
x_line = np.linspace(-20,35)
ug = np.zeros(6)

frequenz = 299792458 * 10**9 / wlaenge
 

for i in range(0, 6):
    werte = csv_read("csv/" + str(farbe[i]) +  ".csv")

    xdata = np.zeros(n[i]-1)
    ydata = np.zeros(n[i]-1) 

    for j in range(0, n[i]-1):
        xdata[j] = float(werte[j+1][1])
        ydata[j] = np.sqrt(float(werte[j+1][0]))
        
    x0 =  float(werte[11][1])
    y0 =  np.sqrt(float(werte[11][0]))
    x_line = np.linspace(np.amin(xdata), np.amax(xdata))
    plt.figure(i)
    plt.plot(xdata, ydata, "rx", label="Messwerte")
    plt.plot(x0, y0, "r.", label="Nicht beachtete Nullmessung")
    popt, pcov = curve_fit(func, xdata, ydata)
    a[i] = popt[0]
    a_err[i] = np.sqrt(pcov[0,0])
    b[i] = popt[1]
    b_err[i] = np.sqrt(pcov[1,1])

    plt.plot(x_line, func(x_line, *popt), "b-", label="Fit")
    #a_i sind D/U_d
    #b_i sind nur Korrekturkoeffizienten
    print("a" + str(i + 1) + " = " + str(popt[0]) + "+/-" + str(np.sqrt(pcov[0,0])))
    print("b" + str(i + 1) + " = " + str(popt[1]) + "+/-" + str(np.sqrt(pcov[1,1])))
    
    plt.xlabel(r"$U_\text{g}$ / V")
    plt.ylabel(r"$\sqrt{I}$ / $\sqrt{\symup{A}}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("build/" + str(farbe[i]) + ".pdf")
ag = unp.uarray(a, a_err)
bg = unp.uarray(b, b_err)

ug = (- bg / ag)
print(str(farbe) + str(ug))
err = unp.std_devs(ug)

x_line2 = np.linspace(np.amin(frequenz), np.amax(frequenz))
plt.figure(6)
plt.errorbar(frequenz, unp.nominal_values(ug), fmt="rx", label="Messwerte", yerr=err)
popt, pcov = curve_fit(func, frequenz, unp.nominal_values(ug)) 

print("------------------------------------")
##a ist die Steigung, b ein Korrekturkoeffizient
print("a7 = " + str(popt[0]) + "+/-" + str(np.sqrt(pcov[0,0])))
print("b7 = " + str(popt[1]) + "+/-" + str(np.sqrt(pcov[1,1])))

plt.plot(x_line2, func(x_line2, *popt), "b-", label="Fit")
#plt.xlabel(r"$f$ / Hz")
#plt.ylabel(r"$U_\text{g}}$ / V")
plt.legend()
plt.tight_layout()
plt.savefig("build/plot_ug.pdf")

with open("build/plots.check", "w") as f:
    f.write("Nur eine Überprüfungsdatei!")
    