import numpy as np
import matplotlib.pyplot as plt
import math

N = 856
M1 = 25
M2 = 25
M = M1 * M2
radio = 6
coef_aprendizaje = 0.1


filename = "../tp2_training_dataset.csv"

entrada = np.genfromtxt(filename, delimiter=',')

W = np.random.randn(M,N)
dW = np.zeros((M,N))
y = np.zeros((M1,M2))
x = np.zeros((1,N))
cant_categorias = 9

matriz_resultados = np.zeros((M1,M2,cant_categorias))
matriz_colores = np.zeros((M1, M2))

cant_rep = 1
num = 1
maximo_rep = 10
while cant_rep < maximo_rep:
	num_fila = 0
	coef_aprendizaje = 1/(float(cant_rep**2))
	#sigma = (float(M2)/2) * 1/(float(cant_rep**3))
	#sigma = (float(M2)/2) * 1/(float(cant_rep))
	sigma_ratio = float(cant_rep) / math.log(float(M2)/2)
	sigma = (float(M2)/2) * math.exp(-(float(cant_rep-1)/sigma_ratio))
	#print "------sigma: "+str(sigma)


	for fila in entrada:
		dW[:] = np.zeros((M,N))
		x[:] = np.array([fila[1:N+1]])
		print "-----x-----"
		print x
		if np.std(x) != 0:
			print "--media: "+str(np.median(x))
			print "--std: "+str(np.std(x))
			x[:] = (x - np.median(x)) / np.std(x)

		print "-----x-----"
		print x

		categoria = fila[0]-1

		yp = np.zeros((M,1))

		yp.T[:] = np.array([np.sum(x - W, axis=-1)])

		y = ((yp == np.min(yp))*1).reshape(M1,M2)

		ganadora = np.nonzero(y)

		if cant_rep == maximo_rep-1:
			#print str(ganadora[0][0])+", "+str(ganadora[1][0])+" -> "+str(categoria)
			matriz_resultados[ganadora[0][0],ganadora[1][0],categoria] += 1

		print str(cant_rep)+" "+str(num_fila)+" -> "+str(ganadora[0][0])+", "+str(ganadora[1][0])


		#...........raro < j / m2, j mod m2 >..............
		P = np.zeros((M,2))
		for j in xrange(0,M):
			P[j,0] = j / M2
			P[j,1] = j % M2

		j_ganadora = ganadora[0][0] * M2 + ganadora[1][0]

		D = np.zeros((M))
		for j in xrange(0,M):
			D[j] = math.exp(-( np.sum((P[j]-P[j_ganadora])**2) ) / (2 * sigma**2))
			dW[j,:] = coef_aprendizaje * D[j] * (x - W[j,:])


		#...........escalones..............
		#y_desde = ganadora[0][0]-radio
		#y_hasta = ganadora[0][0]+radio
		#x_desde = ganadora[1][0]-radio
		#x_hasta = ganadora[1][0]+radio
#
#		#for i in xrange(y_desde,y_hasta+1):
#		#	for j in xrange(x_desde,x_hasta+1):
#		#		distancia = math.fabs(ganadora[0][0] - i)
#		#		if distancia > math.fabs(ganadora[1][0] - j):
#		#			distancia = math.fabs(ganadora[1][0] - j)
		#		dW[(i % M1) * M2 + (j % M2)] = coef_aprendizaje * (x - W[(i % M1) * M2 + (j % M2)]) * (1 - distancia/radio)
				#print "---acomoda -> "+str(i % M1)+", "+str(j % M2)+" ->"+str((i % M1) * M2 + (j % M2))

		W += dW

		num_fila += 1
		num += 1

#	radio -= cant_rep % 10
#	if radio < 1:
#		radio = 1
	cant_rep += 1


print matriz_resultados

for i in xrange(0,M1):
	for j in xrange(0,M2):
		if matriz_resultados[i,j,np.argmax(matriz_resultados[i,j])] == 0:
			matriz_colores[i,j] = 0
		else :
			matriz_colores[i,j] = np.argmax(matriz_resultados[i,j])+1

print matriz_colores

plt.matshow(matriz_colores)
plt.show()