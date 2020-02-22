from pandas import read_csv
import numpy as np
from scipy.optimize import curve_fit as solve

class Correct:
    
    def __init__(self): 
        df = read_csv('pso.csv', header=None)
        data = np.split(df, [1], axis=1)
        times = data[0].values.tolist()
        best_positions = data[1].values.tolist()
        a = []
        b = []
        c = []
        
        print(best_positions[2])
        modes = [1, 2, 3]

        for line in data:
            f0, f1, f2 = self.find_coefs(self.F, best_positions, modes)
            a.append(f0)
            b.append(f1)
            c.append(f2)

        print('a: ', a, 'a: ', b, 'c: ', c)

    def F(self, m, c0, c1, c2):
        return (c0**2 + (m * c1)**2)**0.5 - m * c2

    def find_coefs(self, f, freqs, modes):        
        popt, pconv = solve(self.F, modes, freqs, bounds=(
            [40, -10, -500], [300, 100, 500]), method='trf')
        perr = np.sqrt(np.diag(pconv))
        x, y, z = popt

        return x, y, z


if __name__ == '__main__':
    Correct()