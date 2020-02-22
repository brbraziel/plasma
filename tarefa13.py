
from scipy import io
from scipy import interpolate
from scipy.optimize import curve_fit as solve
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import math
from functools import partial

class DadosMatriz: 
    def __init__(self): 
        dados = {}
        freqs = {}
        times = {}
        ynew = {}
        freqsoma = []
        lin = col = 0
        i = 0
        t_min = 1e300
        t_max = 0
        pontos = io.loadmat("saw_ca3.mat")
        pares = pontos['XYs'][0][:-3]
        arraysoma = []
        aux = 0

        for par in pares:
            x, y = par.transpose()
            self.y = y
            self.x = x
            times[i] = x
            freqs[i] = y
            for t in range(0, len(times[i])): #PEGAR O MENOR TEMPO
                if(times[i][t] < t_min):
                    t_min = times[i][t]
                t += 1     
            for t in range(0, len(times[i])): #PEGAR O MAIOR TEMPO
                if(times[i][t] > t_max):
                    t_max = times[i][t]
                t += 1      
            i += 1

        divisoes = 50 #trocar por 50. ELA DIMINUI O TEMPO DE EXECUÇÃO DIMINUINDO O RANGE
        self.xnew = np.linspace(t_min, t_max, divisoes)
        self.matriz = np.zeros((len(freqs),self.xnew.size))
        self.matriz[np.where(self.matriz==0)[0]] = np.nan
        
        for u in range(len(freqs)):
            mask = np.logical_and(min(times[u]) <= self.xnew, self.xnew <= max(times[u]))
            timenew = self.xnew[mask] 
            f = interpolate.interp1d(times[u], freqs[u])
            ynew = f(timenew)
            
            insertion_time = np.argmax(min(times[u]) <= self.xnew)
            self.matriz[u][insertion_time:insertion_time+len(timenew)] = ynew

        self.matriz = self.matriz.transpose()

        m = np.arange(1, 7) #também podem ser as curvas para encontrar o chi quadrado

        self.a = []
        self.b = []
        self.c = []
        self.x = []
        position = 0
        self.new_times = []
        
        for line in self.matriz[:, m-1]:
            position += 1
            freqs = line
            mask = np.where(~np.isnan(line))
            freqs_time = freqs[mask]
            modes_time = m[mask]
            #print(freqs_time, modes_time) FREQUÊNCIAS E MODES DIFERENTES DE NAN
            f0, f1, f2 = self.find_coefs(self.F, freqs_time, modes_time)
            chi = self.chiquadrado(f0, f1, f2, modes_time, position)
            #print('line: ', line, 'f0: ', f0, 'f1: ', f1, 'f2: ', f2, 'chi: ', chi)
            self.a.append(f0)
            self.b.append(f1)
            self.c.append(f2)
            self.x.append(chi)
            
            
        #self.returning_parameters(xnew, a, b)
        #self.plotarConstantes(a, b, c, x, new_times-new_times[0])
        
    def returning_parameters(self):
        return (self.xnew, self.a, self.b, self.c, self.x)

    def count_nans(self, line):        
        isnan = 0
        for row in line:
                if math.isnan(row):
                    isnan += 1
        return isnan

    def find_coefs(self, fun, freqs, modes):        
        popt, pconv = solve(self.F, modes, freqs, bounds=(
            [40, -10, -500], [300, 100, 500]), method='trf')
        perr = np.sqrt(np.diag(pconv))
        x, y, z = popt

        return x, y, z

    def F(self, m, c0, c1, c2):
        return (c0**2 + (m * c1)**2)**0.5 - m * c2

    def chiquadrado(self, a, b, c, curvas, pos):
        resposta = []
        chi = 0

        for m in curvas:
            resposta=np.sqrt((a*a)+(b*b)*(m*m))+m*c
            chi += (self.matriz[pos-1, m-1] - resposta)**2 

        return chi

    def plotarFrequencias(self, xnew):        
        plt.subplot(3, 1, 1)
        plt.plot(xnew, self.matriz, 'o')
        plt.title('Interpolation')
        plt.ylabel('frequency (kHz)')

        plt.subplot(3, 1, 2)
        plt.plot(xnew, self.matriz.sum(1), '.')
        plt.title('Sum')
        plt.ylabel('frequency (kHz)')

        plt.subplot(3, 1, 3)
        plt.plot(xnew, self.matriz.mean(1), 'o')   
        plt.title('Average')
        plt.ylabel('frequency (kHz)')
        plt.xlabel('time (s)')

        plt.show()
    
    def plotarConstantes(self, f0, f1, f2, chi, times):
        plt.subplot(2, 2, 1)
        plt.plot(times, f0)        
        plt.title('f0')

        plt.subplot(2, 2, 2)
        plt.plot(times, f1)        
        plt.title('f1')

        plt.subplot(2, 2, 3)
        plt.plot(times, f2)        
        plt.title('f2')

        plt.subplot(2, 2, 4)
        plt.plot(times, chi)        
        plt.title('chi²')
        
        plt.show()

if __name__ == '__main__':
    data = DadosMatriz()