import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')


N = 856
M = 10

cF = 600

W = np.random.random_sample((N,M))
dW = np.zeros((N,M))
x = np.zeros((1,N))
xp = np.zeros((N,M))
y = np.zeros((1,M))

es_oja = 1
M_oja = np.triu(np.ones((M,M)))
M_sanjer = np.ones((M,M))


#-- lectura de entrada
filename = "../tp2_training_dataset.csv"
entrada = np.genfromtxt(filename, delimiter=',')
np.random.shuffle(entrada)

datos_entrenamiento = entrada[0:cF,:]
datos_entrenamiento[:,1:N] = datos_entrenamiento[:,1:N] / 10
datos_validacion = entrada[cF:,:]
datos_validacion[:,1:N] = datos_validacion[:,1:N] / 10

num_epoca = 1
cant_rep = 1
fact_act = 0.2
error_min = 20
ultimo_error = 20
cant_ultimo_error = 0
while num_epoca < 200 and error_min > 0.000001 and cant_ultimo_error < 5:
	#fact_act -= fact_act / 2
	for fila in datos_entrenamiento:

		fact_act = 1/float(cant_rep)
		
		x[:] = np.array([fila[1:]])

		y[:] = x.dot(W)

		if es_oja == 1:
			xp[:] = W.dot(np.multiply(y.T, M_oja))
		else:
			xp[:] = W.dot(np.multiply(y.T, M_sanjer))


		dW[:] = fact_act * np.multiply((x.T - xp),y)
		
		W[:] = W + dW
		cant_rep += 1

	error_min = 20000

	for i in xrange(0,M-1):
		for j in xrange(i+1,M):
			if error_min > math.fabs(W[:,i].dot(W[:,j])):
				error_min =  math.fabs(W[:,i].dot(W[:,j]))

	print "---epoca "+str(num_epoca)+": "+str(error_min)

	if error_min == ultimo_error:
		cant_ultimo_error += 1
	else:
		cant_ultimo_error = 0
		ultimo_error = error_min
	num_epoca += 1


W = W[:,0:3]

colores_cat = cm.rainbow(np.linspace(0, 1, 10))

for cat in xrange(1,10):
	pos_categoria = np.empty((1,3));
	for fila in datos_entrenamiento:
		if int(fila[0]) == int(cat):
			x[:] = np.array([fila[1:]])
			pos_categoria = np.append(pos_categoria, x.dot(W), axis=0)
	ax.scatter(pos_categoria[:,0], pos_categoria[:,1], zs=pos_categoria[:,2], color=colores_cat[cat])

#fig.savefig("saida_entrenamiento.png")
fig.show()

raw_input()

for cat in xrange(1,10):
	pos_categoria = np.empty((1,3));
	for fila in datos_validacion:
		if int(fila[0]) == int(cat):
			x[:] = np.array([fila[1:]])
			pos_categoria = np.append(pos_categoria, x.dot(W), axis=0)
	ax2.scatter(pos_categoria[:,0], pos_categoria[:,1], zs=pos_categoria[:,2], color=colores_cat[cat])

#fig2.savefig("saida_validacion.png")
fig2.show()

raw_input()