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
        self.entradas = 200
        self.input_file = "../tp2_training_dataset.csv" #--archivo csv de entrada

        #--------------------------------------------
        self.N = 856
        self.M1 = self.dimension
        self.M2 = self.dimension
        self.M = self.M1 * self.M2
        self.W = np.random.random_sample((self.M,self.N))
        self.dW = np.zeros((self.M,self.N))
        self.y = np.zeros((self.M1,self.M2))
        self.x = np.zeros((1,self.N))
        self.cant_categorias = 9


    def entrenar(self):
		num = 1
		entrada = np.genfromtxt(self.input_file, delimiter=',')
		datos_entrenamiento = entrada[0:self.entradas,:]

		epoca_actual = 1
		norma_actual = 0
		norma_anterior = 0
		diferencia_norma = 100

		while epoca_actual < self.cantidad_epocas and diferencia_norma > self.tolerancia_error:
			print "epoca:" +  str(epoca_actual)
			print "diferencia:" +  str(diferencia_norma)
			matriz_resultados = np.zeros((self.M1,self.M2,self.cant_categorias))
			matriz_colores = np.zeros((self.M1, self.M2))
			sigma_o = self.M2/2
			lambd = float(self.cantidad_epocas*len(datos_entrenamiento))/math.log(sigma_o)

			for fila in datos_entrenamiento:
				# va bien
				#sigma = sigma_o * math.exp(-float(epoca_actual)/lambd)
				#coef_aprendizaje = self.learning_rate * math.exp(-float(epoca_actual)/lambd)

				sigma = sigma_o * math.exp(-float(num)/lambd)
				#coef_aprendizaje = self.learning_rate * math.exp(-float(num)/lambd)
				
				#sigma = sigma_o * 1 / (float(epoca_actual)**(1/3) * (1/float(self.cantidad_epocas*len(datos_entrenamiento))*1/4 / 1/4) + 1)
				coef_aprendizaje = self.learning_rate * 0.9
				self.dW[:] = np.zeros((self.M,self.N))
				self.x[:] = np.array([fila[1:self.N+1]])
				categoria = fila[0]-1
				yp = np.zeros((self.M,1))
				yp.T[:] = np.array([np.sum(np.absolute(self.x - self.W), axis=-1)])
				self.y = ((yp == np.min(yp))*1).reshape(self.M1,self.M2)
				ganadora = np.nonzero(self.y)
				matriz_resultados[ganadora[0][0],ganadora[1][0],categoria] += 1

				#...........bmu..............
				D = np.zeros((self.M))
				for i in xrange(0,self.M1):
					for j in xrange(0,self.M2):
						suma = 0
						suma += (i-ganadora[0][0])**2
						suma += (j-ganadora[1][0])**2
						if suma < (sigma**2):
							D[(i * self.M2) + j] = math.exp(- suma / (2 * sigma**2))
							self.dW[(i * self.M2) + j,:] = (coef_aprendizaje * D[(i * self.M2) + j] * (self.x - self.W[(i * self.M2) + j,:])).reshape(self.N)

				self.W += self.dW
				num += 1

			norma_actual = np.linalg.norm(self.W)
			diferencia_norma = abs(norma_actual - norma_anterior)
			norma_anterior = norma_actual

			epoca_actual += 1

			for i in xrange(0,self.M1):
				for j in xrange(0,self.M2):
					if matriz_resultados[i,j,np.argmax(matriz_resultados[i,j])] == 0:
						matriz_colores[i,j] = 0
					else :
						matriz_colores[i,j] = np.argmax(matriz_resultados[i,j])+1


			#if epoca_actual % 20 == 0: 
				#plt.matshow(matriz_colores)
				#plt.show()
				#plt.savefig("mapa_"+str(epoca_actual)+".png")

		plt.matshow(matriz_colores)
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
            self.dimension = float(valor)
        elif clave == 'cant':
            self.entradas = float(valor)
        elif clave == 'input':
            self.input_file = valor


    def importarW(self, filename):
        csv_entrada = csv.reader(open(filename, "rb"), delimiter=',')
        
        return "TODO"


    def exportarW(self, filename):
        csv_salida = csv.writer(open(filename, "wb"))
        csv_salida.writerow([self.W])
        for i in xrange(0,len(self.W)):
            csv_salida.writerow(self.W[i])
       
        return "Mapa exportado con exito!"


    def testing(self):
        #--busca datos del archivo de entrada

        return "TODO"


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
        msg = koh.exportarW(comando[1])
    elif (comando[0] == 'import'):
        msg = koh.importarW(comando[1])
    elif (comando[0] == 'test'):
        koh.testing()
        raw_input("Pulse enter para volver al menu...")
