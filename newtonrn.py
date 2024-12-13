import matplotlib.pyplot as plt
import numpy as np

erro = 10e-4
pontos_iniciais = [[1.2, 1.7], [2, 3], [0, 1], [-1, 0], [0, 0]]
jacobiana = lambda x1, x2: [[6 * x1**2, -2 * x2], [x2**3, 3 * x1 * x2**2 - 1]]
Func = lambda x1, x2: [2 * x1**3 - x2**2 - 1, x1 * x2**3 - x2 - 4]
menos = lambda x: [-i for i in x]

for ponto in pontos_iniciais:
    iteracoes = 0
    x1 = ponto[0]
    x2 = ponto[1]
    x1_vals = [x1]
    x2_vals = [x2]
    while True:
        print(x1, x2)
        iteracoes += 1
        jacob = jacobiana(x1, x2)
        F = Func(x1, x2)
        try:
            v = np.linalg.solve(jacob, menos(F))
        except np.linalg.LinAlgError:
            print("Solução não encontrada")
            break
        x1 += v[0]
        x2 += v[1]
        x1_vals.append(x1)
        x2_vals.append(x2)
        plt.plot(x1_vals, x2_vals, marker='o')
        if np.linalg.norm(v) < erro:
            break
    

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Iterações do Método de Newton')
    plt.grid(True)
    plt.show()

    print(x1, x2)
    print("Iterações: "+ str(iteracoes))
    print("##################")
