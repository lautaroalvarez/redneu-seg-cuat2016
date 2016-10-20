import numpy as np
import matplotlib.pyplot as plt
import math

N = 856
#N = 4
M1 = 15
M2 = 15
M = M1 * M2
radio = 9
coef_aprendizaje_const = 0.1


filename = "../tp2_training_dataset.csv"
#filename = "prueba.csv"

entrada = np.genfromtxt(filename, delimiter=',')
datos_entrenamiento = entrada[0:200,:]
#datos_entrenamiento = np.array([entrada[:]])
datos_entrenamiento[:,1:N+1] = datos_entrenamiento[:,1:N+1] / 10


W = np.random.random_sample((M,N))
dW = np.zeros((M,N))
y = np.zeros((M1,M2))
x = np.zeros((1,N))
cant_categorias = 9



num_etapa = 1
num = 1
maximo_etapas = 30
while num_etapa < maximo_etapas:
	num_fila = 0
	#coef_aprendizaje = 1/(float(num_etapa))
	#sigma = (float(M2)/2) * 1/(float(num_etapa**3))
	#sigma = (float(M2)/2) * 1/(float(num_etapa))
	#sigma_ratio = float(num_etapa) / math.log(float(M2)/2)
	#sigma = (float(M2)/2) * math.exp(-(float(num_etapa-1)/sigma_ratio))
	#print "------sigma: "+str(sigma)
	
	#sigma_ratio = 1 / float(num_etapa)
	#sigma = (M2/2) * math.exp(-4*num_etapa/maximo_etapas)
	#sigma = (M2/2) * (float(maximo_etapas-num_etapa))/maximo_etapas

#	sigma_o = M2/2
#	lambd = float(maximo_etapas)/math.log(sigma_o)
#	sigma = sigma_o * math.exp(-float(num_etapa)/lambd)
#	coef_aprendizaje = coef_aprendizaje_const * math.exp(-float(num_etapa)/lambd)
	
	matriz_resultados = np.zeros((M1,M2,cant_categorias))
	matriz_colores = np.zeros((M1, M2))
	
	sigma_o = M2/2
	lambd = float(maximo_etapas*len(datos_entrenamiento))/math.log(sigma_o)
	for fila in datos_entrenamiento:
		
		sigma = sigma_o * math.exp(-float(num)/lambd)
		coef_aprendizaje = coef_aprendizaje_const * math.exp(-float(num)/lambd)
		
		dW[:] = np.zeros((M,N))
		x[:] = np.array([fila[1:N+1]])

		categoria = fila[0]-1

		yp = np.zeros((M,1))

		yp.T[:] = np.array([np.sum(np.absolute(x - W), axis=-1)])

		y = ((yp == np.min(yp))*1).reshape(M1,M2)

		ganadora = np.nonzero(y)

		#if num_etapa == maximo_etapas-1:
			#print str(ganadora[0][0])+", "+str(ganadora[1][0])+" -> "+str(categoria)
		matriz_resultados[ganadora[0][0],ganadora[1][0],categoria] += 1

		print str(num_etapa)+" "+str(num_fila)+" -> "+str(ganadora[0][0])+", "+str(ganadora[1][0])


		#...........raro < j / m2, j mod m2 >..............
#		P = np.zeros((M,2))
#		for j in xrange(0,M):
#			P[j,0] = j / M2
#			P[j,1] = j % M2
#
#		j_ganadora = (ganadora[0][0] * M2) + ganadora[1][0]
#
#		D = np.zeros((M))
#		for j in xrange(0,M):
#			suma = 0
#			suma += (P[j,0]-P[j_ganadora,0])**2
#			suma += (P[j,1]-P[j_ganadora,1])**2
#			D[j] = math.exp(- suma / (2 * sigma**2))
#			dW[j,:] = (coef_aprendizaje * D[j] * (x - W[j,:])).reshape(N)


		#...........bmu..............
		D = np.zeros((M))
		for i in xrange(0,M1):
			for j in xrange(0,M2):
				suma = 0
				suma += (i-ganadora[0][0])**2
				suma += (j-ganadora[1][0])**2
				if suma < (sigma**2):
					D[(i * M2) + j] = math.exp(- suma / (2 * sigma**2))
					dW[(i * M2) + j,:] = (coef_aprendizaje * D[(i * M2) + j] * (x - W[(i * M2) + j,:])).reshape(N)


		#...........escalones..............
#		y_desde = ganadora[0][0]-radio
#		y_hasta = ganadora[0][0]+radio
#		x_desde = ganadora[1][0]-radio
#		x_hasta = ganadora[1][0]+radio

#		for i in xrange(y_desde,y_hasta+1):
#			for j in xrange(x_desde,x_hasta+1):
#				distancia = math.fabs(ganadora[0][0] - i)
#				if distancia > math.fabs(ganadora[1][0] - j):
#					distancia = math.fabs(ganadora[1][0] - j)
#				dW[(i % M1) * M2 + (j % M2)] = coef_aprendizaje * (x - W[(i % M1) * M2 + (j % M2)]) * (1 - distancia/radio)
				#print "---acomoda -> "+str(i % M1)+", "+str(j % M2)+" ->"+str((i % M1) * M2 + (j % M2))

		W += dW

		num_fila += 1
		num += 1

#	radio -= num_etapa % 3
#	if radio < 1:
#		radio = 1
	num_etapa += 1

	#print matriz_resultados

	for i in xrange(0,M1):
		for j in xrange(0,M2):
			if matriz_resultados[i,j,np.argmax(matriz_resultados[i,j])] == 0:
				matriz_colores[i,j] = 0
			else :
				matriz_colores[i,j] = np.argmax(matriz_resultados[i,j])+1

	print matriz_colores

	plt.matshow(matriz_colores)
	#plt.show()
	plt.savefig("mapa_"+str(num_etapa)+".png")