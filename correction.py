from pandas import read_csv
import numpy as np
from scipy.optimize import curve_fit as solve
import tarefa19 as tratante
import tarefa13 as t

class Correct:
    
    def __init__(self): 
        df = read_csv('pso.csv', header=None)
        positions_array = np.split(df, [1, 4], axis=1)
        chis_array = np.split(df, [4], axis=1)
        times = positions_array[0].values.tolist()
        self.delta = 2
        position = 0

        self.orig_data = tratante.Tratamento()

        self.best_positions = positions_array[1].values
        self.chis = chis_array[1].values
        N = len(self.best_positions)
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        chis = []
        
        for line in range(N):
            position += 1
            mask = np.where(~np.isnan(self.orig_data.matriz[line]))
            freqs_time = self.orig_data.matriz[line][mask]
            modes_time = self.orig_data.m[mask]
            f0, f1, f2 = self.find_coefs(self.F, freqs_time, modes_time, self.best_positions[line])
            a[line] = f0
            b[line] = f1
            c[line] = f2

            self.chi = t.DadosMatriz.chiquadrado(self, self.orig_data.matriz, f0, f1, f2, modes_time, position)
            chis.append(self.chi)

        print('a: ', a, '\nb: ', b, '\nc: ', c, '\nchis:', chis)
        t.DadosMatriz.plotarConstantes(self, a, b, c, chis, times)


    def F(self, m, c0, c1, c2):
        return (c0**2 + (m * c1)**2)**0.5 - m * c2

    def find_coefs(self, f, freqs, modes, limits):
        """freqs são as frequências medidas
        limits são os limites dados pelo pso"""        
        popt, pconv = solve(f, modes, freqs, bounds=(
            limits-self.delta, limits+self.delta), method='trf')
        perr = np.sqrt(np.diag(pconv))
        x, y, z = popt

        return x, y, z

if __name__ == '__main__':
    core = Correct()