import numpy as np
import matplotlib.pyplot as plt
import math
import csv, sys, math, os


class kohonen:
	def __init__(self):
		#--------------------------------------------
		#------Parametros
		self.learning_rate = 0.999
		self.tolerancia_error = 0.00001
		self.cantidad_epocas = 100000
		self.dimension = 20
		self.entradas = 300
		self.input_file = "../tp2_training_dataset.csv" #--archivo csv de entrada

		self.data_entrenamiento = np.empty((1,1))
		self.data_validacion = np.empty((1,1))

		#--------------------------------------------
		self.N = 856
		self.M1 = self.dimension
		self.M2 = self.dimension
		self.M = self.M1 * self.M2
		self.W = np.random.random_sample((self.M,self.N))
		self.cant_categorias = 9

		self.actualizarDataSet()
		
		self.Mres = np.zeros((self.M1,self.M2)) #--resultado con las categorias asignadas


	def actualizarDataSet(self):
		#--lectura de archivo de entrada
		data_input = np.genfromtxt(self.input_file, delimiter=',')
		#--preprocesamiento
		data_input[:,1:] = data_input[:,1:] / 10
		
		#--cambiamos el orden del dataset
		np.random.shuffle(data_input)
		self.data_entrenamiento = data_input[0:self.entradas,:]
		self.data_validacion = data_input[self.entradas+1:,:]


	def entrenar(self):
		num = 1

		epoca_actual = 1
		norma_actual = 0
		norma_anterior = 0
		diferencia_norma = 100
		
		dW = np.zeros((self.M,self.N))
		y = np.zeros((self.M1,self.M2))
		x = np.zeros((1,self.N))
		yp = np.zeros((self.M,1))

		self.Mres[:] = 0;

		while epoca_actual < self.cantidad_epocas and diferencia_norma > self.tolerancia_error:
			print "epoca:" +  str(epoca_actual)
			print "diferencia:" +  str(diferencia_norma)
			matriz_resultados = np.zeros((self.M1,self.M2,self.cant_categorias))
			sigma_o = self.M2/2
			lambd = float(self.cantidad_epocas*len(self.data_entrenamiento))/math.log(sigma_o)
			coef_aprendizaje = epoca_actual ** (-1)
			sigma = sigma_o * epoca_actual ** (-1.0/3.0)
			coef_aprendizaje =self.learning_rate / (1 + epoca_actual * 0.5 * self.learning_rate)

			for fila in self.data_entrenamiento:
				# va bien
				#sigma = sigma_o * math.exp(-float(epoca_actual)/lambd)
				#coef_aprendizaje = self.learning_rate * math.exp(-float(epoca_actual)/lambd)

				#sigma = sigma_o * math.exp(-float(num)/lambd)
				#coef_aprendizaje = self.learning_rate * math.exp(-float(num)/lambd)
				
				#sigma = sigma_o * 1 / (float(epoca_actual)**(1/3) * (1/float(self.cantidad_epocas*len(self.data_entrenamiento))*1/4 / 1/4) + 1)
				#coef_aprendizaje = self.learning_rate * 0.9
				dW[:] = np.zeros((self.M,self.N))
				x[:] = np.array([fila[1:self.N+1]])
				categoria = fila[0]-1
				yp.T[:] = np.array([np.sum(np.absolute(x - self.W), axis=-1)])
				y = ((yp == np.min(yp))*1).reshape(self.M1,self.M2)
				ganadora = np.nonzero(y)
				matriz_resultados[ganadora[0][0],ganadora[1][0],categoria] += 1


				D = np.zeros((self.M))
				for i in xrange(0,self.M1):
					for j in xrange(0,self.M2):
						suma = 0
						suma += (i-ganadora[0][0])**2
						suma += (j-ganadora[1][0])**2
						if suma < (sigma**2):
							D[(i * self.M2) + j] = math.exp(- suma / (2 * sigma**2))
							dW[(i * self.M2) + j,:] = (coef_aprendizaje * D[(i * self.M2) + j] * (x - self.W[(i * self.M2) + j,:])).reshape(self.N)

				self.W += dW
				num += 1

			norma_actual = np.linalg.norm(self.W)
			diferencia_norma = abs(norma_actual - norma_anterior)
			norma_anterior = norma_actual

			epoca_actual += 1

			for i in xrange(0,self.M1):
				for j in xrange(0,self.M2):
					if matriz_resultados[i,j,np.argmax(matriz_resultados[i,j])] == 0:
						self.Mres[i,j] = 0
					else :
						self.Mres[i,j] = np.argmax(matriz_resultados[i,j])+1


			if epoca_actual % 20 == 0: 
				plt.matshow(self.Mres)
				#plt.show()
				plt.savefig("mapa_"+str(epoca_actual)+".png")

		plt.matshow(self.Mres)
		#plt.show()
		plt.savefig("mapa_"+str(epoca_actual)+".png")


		return "Fin entrenamiento"

	def cambiar_valor(self, clave, valor):
		if clave == 'lr':
			self.learning_rate = float(valor)
		elif clave == 'tol':
			self.tolerancia_error = float(valor)
		elif clave == 'cep':
			self.cantidad_epocas = float(valor)
		elif clave == 'dim':
			#--modifica dimensiones en matrices
			tamano_viejo = self.dimension
			self.dimension = int(valor)
			self.M1 = self.dimension
			self.M2 = self.dimension
			self.M = self.M1 * self.M2
			if tamano_viejo < self.dimension:
				self.W = np.append(self.W, np.random.randn(self.M - tamano_viejo**2, self.N), axis=1)
				self.Mres = np.append(self.Mres, np.random.randn(self.dimension - tamano_viejo, self.N), axis=1)
			else:
				self.W = self.W[0:self.M,0:self.N]
				self.Mres = self.Mres[0:self.dimension,0:self.dimension]
		elif clave == 'cant':
			self.entradas = float(valor)
		elif clave == 'input':
			self.input_file = valor
			self.actualizarDataSet()

	def importarRed(self, filename):
		csv_entrada = csv.reader(open(filename, "rb"), delimiter=',')
		
		cont = 0
		for row in csv_entrada:
			if cont == 0:
				#--modifica dimensiones en matrices
				tamano_viejo = self.dimension
				self.dimension = int(row[0])
				self.M1 = self.dimension
				self.M2 = self.dimension
				self.M = self.M1 * self.M2
				if tamano_viejo < self.dimension:
					self.W = np.append(self.W, np.random.randn(self.M - tamano_viejo**2, self.N), axis=1)
					self.Mres = np.append(self.Mres, np.random.randn(self.dimension - tamano_viejo, self.N), axis=1)
				else:
					self.W = self.W[0:self.M,0:self.N]
					self.Mres = self.Mres[0:self.dimension,0:self.dimension]
				print self.W.shape
				print self.Mres.shape
			elif cont <= self.M:
				print "entra a w: "+str(cont)
				self.W[cont-1,:] = row
			else:
				print "entra a Mres: "+str(cont)+" -> "+str(cont-self.M)
				self.Mres[cont-self.M-1,:] = row
			cont += 1
		return "Red importada con exito!"

	def exportarRed(self, filename):
		csv_salida = csv.writer(open(filename, "wb"))
		csv_salida.writerow([self.dimension])
		for fila in self.W:
			csv_salida.writerow(fila)
		for fila in self.Mres:
			csv_salida.writerow(fila)
		return "Red exportada con exito!"


	def testing(self):
		cant_bien = 0
		cant_vacio = 0

		y = np.zeros((self.M1,self.M2))
		x = np.zeros((1,self.N))
		yp = np.zeros((self.M,1))
		
		for fila in self.data_validacion:
			x[:] = np.array([fila[1:self.N+1]])
			categoria = fila[0]

			yp.T[:] = np.array([np.sum(np.absolute(x - self.W), axis=-1)])
			y = ((yp == np.min(yp))*1).reshape(self.M1,self.M2)
			ganadora = np.nonzero(y)
		
			if self.Mres[ganadora[0][0]][ganadora[1][0]] == categoria:
				#--cayo en su categoria
				cant_bien += 1
			elif self.Mres[ganadora[0][0]][ganadora[1][0]] == 0:
				cant_vacio += 1

		print "Resultado Validacion:"
		print "      "+str(cant_bien)+" bien"
		print "      "+str(cant_vacio)+" sin determinar"
		print "      "+str(len(self.data_validacion)-cant_bien)+" mal"

		return


def mostrar_menu(koh, msg):
	os.system('clear');
	print "MENU DEL EJERCICIO2:\n"
	print "Parametros activos:"
	print "  - learning rate:             (lr)     " + str(koh.learning_rate)
	print "  - tolerancia de error:       (tol)    " + str(koh.tolerancia_error)
	print "  - cantidad epocas:           (cep)    " + str(koh.cantidad_epocas)
	print "  - dimension del mapa:        (dim)    " + str(koh.dimension)
	print "  - cantidad de datos (train): (cant)   " + str(koh.entradas)
	print "  - input file (input) " + str(koh.input_file)
	print "\n"
	print "help -> para ver los comandos validos y su modo de uso"
	print "exit -> terminar la ejecucion del programa\n"
	if len(msg) > 0:
		print "Respuesta:  "+str(msg)
	else:
		print ""
	print ""



def mostrar_ayuda():
	os.system('clear');
	print "AYUDA DEL EJERCICIO2:\n"
	print "Acciones validas:\n"
	print "  - Importar un mapa de Kohonen"
	print "         Descripcion: importa un mapa desde un archivo"
	print "         Uso: import nombre_archivo.extension"
	print "         Ejemplo: import mapaOK.in"
	print ""
	print "  - Exportar un mapa"
	print "         Descripcion: exporta el mapa actual hacia un archivo"
	print "         Uso: export nombre_archivo.extension"
	print "         Ejemplo: export mapa_train.in"
	print ""
	print "  - Cambiar parametro"
	print "         Descripcion: modifica el valor de un parametro"
	print "         Uso: change codigo_parametro nuevo_valor"
	print "         Ejemplo: change lr 0.001"
	print ""
	print "  - Comenzar entrenamiento"
	print "         Descripcion: ejecuta el entrenamiento con los parametros guardados"
	print "         Uso: train"
	print ""
	print "  - Testear Red"
	print "         Descripcion: testea la respuesta de la red a los datos de validacion"
	print "         Uso: test"
	print ""

koh = kohonen()

salir = 0
msg = ""


while (not salir):
	mostrar_menu(koh, msg)
	msg = ""
	comando = raw_input("Ingrese un comando: ");
	comando = comando.split(" ")
	if (comando[0] == 'exit'):
		salir = 1
	elif (comando[0] == 'help'):
		mostrar_ayuda()
		raw_input("Pulse enter para volver al menu...")
	elif (comando[0] == 'train'):
		msg = koh.entrenar()
		raw_input("Pulse enter para volver al menu...")
	elif (comando[0] == 'change'):
		koh.cambiar_valor(comando[1], comando[2])
	elif (comando[0] == 'export'):
		msg = koh.exportarRed(comando[1])
	elif (comando[0] == 'import'):
		msg = koh.importarRed(comando[1])
	elif (comando[0] == 'test'):
		koh.testing()
		raw_input("Pulse enter para volver al menu...")
