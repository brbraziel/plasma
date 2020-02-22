import random
from scipy import io
from scipy import interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import DataFrame

class Tratamento:
    def __init__(self):
        pontos = io.loadmat("saw_ca3.mat")
        pares = pontos['XYs'][0][:-3]
        self.m_atual = []
        self.freq_atual = []
        self.modes(pontos, pares)

    def modes(self, pontos, pares):
        freqs = {}
        times = {}
        ynew = {}
        freqsoma = []
        lin = col = 0
        i = 0
        t_min = 1e300
        t_max = 0        
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

        self.m = np.arange(1, 7) #também podem ser as curvas para encontrar o chi quadrado

    def get_position(self, position):    
        line = self.matriz[position, :]
        mask = np.where(~np.isnan(line))
        m_new = self.m[mask]
        freq_new = self.matriz[position, :][mask]
        return m_new, freq_new

    def count_nans(self, line):        
        isnan = 0
        for row in line:
            if math.isnan(row):
                isnan += 1
        return isnan

    def get_times(self):
        return self.xnew

class PSO:
    #function that models the problem
    def fitness_function(self, position, ms, freqs):
        fitness = 0
        for mode in range(len(ms)):
            fitness += abs((position[0]**2 + (ms[mode] * position[1])**2)**0.5 - ms[mode] * position[2] - freqs[mode])
        return fitness

    def __init__(self):        
        tratar = Tratamento()
        self.times = tratar.get_times()
        self.best_positions = []
        pos = 0
        self.chis = []
        x1 = []
        x2 = []
        x3 = []

        for time in self.times:
            W = 0.7
            c1 = 0.7
            c2 = 0.9
            target = 1
            n_iterations = 500
            target_error = 1e-6
            n_particles = 70

            ms,freqs = tratar.get_position(pos)

            particle_position_vector = [np.array([(1+0.1*random.random())*50,  random.random()*20, random.random()*10]) for _ in range(n_particles)]
            #print(particle_position_vector)
            pbest_position = particle_position_vector
            pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])
            self.gbest_fitness_value = float('inf')
            gbest_position = np.array([float('inf'), float('inf')])

            velocity_vector = ([np.array([0, 0, 0]) for _ in range(n_particles)])
            iteration = 0

            while iteration < n_iterations:
                for i in range(n_particles):
                    self.fitness_candidate = self.fitness_function(particle_position_vector[i], ms, freqs)
                    #print(self.fitness_cadidate, ' ', particle_position_vector[i])
                    
                    if(pbest_fitness_value[i] > self.fitness_candidate):
                        pbest_fitness_value[i] = self.fitness_candidate
                        pbest_position[i] = particle_position_vector[i]

                    if(self.gbest_fitness_value > self.fitness_candidate):
                        self.gbest_fitness_value = self.fitness_candidate #gbest é o chi quadrado
                        gbest_position = particle_position_vector[i]

                if(abs(self.gbest_fitness_value - target) < target_error):
                    break
                
                for i in range(n_particles):
                    new_velocity = (W*velocity_vector[i]) + (c1*random.random()) * (pbest_position[i] - particle_position_vector[i]) + (c2*random.random()) * (gbest_position-particle_position_vector[i])
                    new_position = new_velocity + particle_position_vector[i]
                    particle_position_vector[i] = new_position

                iteration = iteration + 1
            gbest_position = np.split(gbest_position, 3, axis=0)
            x1.append(gbest_position[0][0])
            x2.append(gbest_position[1][0])
            x3.append(gbest_position[2][0])
            print(x1, x2, x3)
            
            pos = pos + 1
            
            self.chis.append(self.gbest_fitness_value)
            
            print("The best positions for time", time ,"are", gbest_position, "in iteration number", iteration, "with fitness", self.gbest_fitness_value)
        
        
        data = {'Time': self.times,
            'X1': x1,
            'X2': x2,
            'X3': x3,
        }
        df = DataFrame(data)

        df.to_csv(r'pso.csv', index=None, header=None)

    def returning_parameters(self):
        self.best_positions = np.array(self.best_positions)
        return (self.times, self.best_positions[:,0], self.best_positions[:,1], self.best_positions[:,2], self.chis)

if __name__ == '__main__':
    pso = PSO()
