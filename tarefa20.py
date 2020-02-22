import matplotlib.pyplot as plt
import numpy as np
import tarefa19
import tarefa13

class Plotagem:
    def __init__(self):
        tenho_tempo=False
        if tenho_tempo:
            data = tarefa13.DadosMatriz()
            (times_curve, f0_curve, f1_curve, f2_curve, chi_curve) = tarefa13.DadosMatriz().returning_parameters()
            np.save("curve_fit",(times_curve, f0_curve, f1_curve, f2_curve, chi_curve))
            data = tarefa19.PSO()
            (times_pso, f0_pso, f1_pso, f2_pso, chi_pso) = tarefa19.PSO().returning_parameters()
            np.save("pso",(times_pso, f0_pso, f1_pso, f2_pso, chi_pso))
        else:
            times_curve, f0_curve, f1_curve, f2_curve, chi_curve=np.load("curve_fit.npy")
            times_pso, f0_pso, f1_pso, f2_pso, chi_pso=np.load("pso.npy")
        
        print("PSO:")
        print("time:", times_pso, "f0:", f1_pso, "f1:", f1_pso, "f2:", f2_pso)
        print("CURVE FIT:")
        print("time:", times_curve, "f0:", f0_curve, "f1:", f1_curve, "f2:", f2_curve)

        plt.subplot(4, 1, 1)
        plt.plot(times_pso, f0_pso, label='PSO')    
        plt.plot(times_curve, f0_curve, label='Curve Fit')
        plt.legend(loc='best')
        plt.title('f0')
        plt.ylim(50,61)

        plt.subplot(4, 1, 2)
        plt.plot(times_pso, f1_pso, label='PSO') 
        plt.plot(times_curve, f1_curve, label='Curve Fit')   
        plt.legend(loc='best')     
        plt.title('f1')
        plt.ylim(-2.2,19.4)

        plt.subplot(4, 1, 3)
        plt.plot(times_pso, f2_pso, label='PSO') 
        plt.plot(times_curve, f2_curve, label='Curve Fit')   
        plt.legend(loc='best')     
        plt.title('f2')
        plt.ylim(-2.1,6)

        plt.subplot(4, 1, 4)
        plt.plot(times_pso, chi_pso, label='PSO') 
        plt.plot(times_curve, chi_curve, label='Curve Fit')   
        plt.legend(loc='best')     
        plt.title('Chi²')
        plt.ylim(-100,12000)

        plt.show()

if __name__ == '__main__':
    plot = Plotagem()