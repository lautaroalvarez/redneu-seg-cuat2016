import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math, sys, csv
from mpl_toolkits.mplot3d import Axes3D
#from sklearn import preprocessing

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')


N = 856
M = 3

cF = 600

W = np.random.random_sample((N,M))
dW = np.zeros((N,M))
x = np.zeros((1,N))
xp = np.zeros((N,M))
y = np.zeros((1,M))

es_oja = 1
if len(sys.argv) > 1:
	es_oja = int(sys.argv[1])

M_sanjer = np.triu(np.ones((M,M)))
M_oja = np.ones((M,M))


#-- lectura de entrada
filename = "../tp2_training_dataset.csv"
entrada = np.genfromtxt(filename, delimiter=',')
np.random.shuffle(entrada)

#entrada[:,1:] = entrada[:,1:] + 2

#entrada[:,1:] = preprocessing.scale(entrada[:,1:])
#entrada[:,1:] = preprocessing.normalize(entrada[:,1:], norm='l1')

#print np.amax(entrada[:,1:])
#print np.mean(entrada[:,1:])
#print np.std(entrada[:,1:])
#print "-------------------"

#for fila in entrada[:,1:]:
#	for i in xrange(0,len(fila)):
#		if fila[i] > 1 or fila[i] < -1:
#			print fila[i]

#for i in xrange(1,N+1):
#	print "----------i: "+str(i)
#	print np.median(entrada[:,i])
#	print np.std(entrada[:,i])
#	entrada[:,i] = (entrada[:,i] - np.median(entrada[:,i])) / np.std(entrada[:,i])
#entrada[:,1:] = (entrada[:,1:] - np.median(entrada[:,1:])) / np.std(entrada[:,1:])

#print np.amax(entrada[:,1:])

#entrada[:,1:] = entrada[:,1:] / 10
datos_entrenamiento = entrada[:cF,:]
datos_validacion = entrada[cF:,:]

fact_act = 0.2
cota_error = 0.00001
cantidad_epocas = 2000

csv_salida = csv.writer(open("salida.csv", "wb"))
csv_salida.writerow([filename, "1/t2", cantidad_epocas, cota_error, es_oja])
csv_salida.writerow(["Grafico de convergencia de la normal", "Error", "Epoca"])
csv_salida.writerow(["Error"])

num_epoca = 1
cant_rep = 1
error = 20
while num_epoca < cantidad_epocas and error > cota_error:
	#fact_act -= fact_act / 2
	for fila in datos_entrenamiento:

		#fact_act = 1/(float(cant_rep)**0.5)
		fact_act = 1/float(cant_rep)
		
		x[:] = np.array([fila[1:]])

		y[:] = x.dot(W)

		if es_oja == 1:
			xp[:] = W.dot(np.multiply(y.T, M_oja))
		else:
			xp[:] = W.dot(np.multiply(y.T, M_sanjer))

		dW[:] = fact_act * np.multiply((x.T - xp),y)
		
		#print dW

		W[:] = W + dW
		cant_rep += 1

	error = np.linalg.norm(W.T.dot(W) - np.identity(M))

	if num_epoca % 20 == 0:
		print "---epoca "+str(num_epoca)+": "+str(error)

	csv_salida.writerow([num_epoca, error])

	num_epoca += 1

csv_salida.writerow([])

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