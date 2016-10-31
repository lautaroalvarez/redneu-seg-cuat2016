import csv, sys, math, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import graficador

class hebbian:
    def __init__(self):
        #--------------------------------------------
        #------Parametros
        self.tolerancia_error = 0.0001
        self.cantidad_epocas = 2000

        self.input_file = "../tp2_training_dataset.csv" #--archivo csv de entrada
        #self.input_file = "prueba.csv"
        self.output_file_train = "ej1_train.sal" #--archivo csv de salida de training
        self.output_img_train = "ej1_train.png" #--archivo csv de salida de validacion
        self.output_img_valid = "ej1_valid.png" #--archivo csv de salida de validacion

        self.data_entrenamiento = np.empty((1,1))
        self.data_validacion = np.empty((1,1))
        
        self.oja = 0

        #--------------------------------------------

        self.tamano_entrada = 856 #--dimension de la entrada
        self.tamano_salida = 3 #--dimension de la salida
        self.dimension_final = 3 #--dimension final de la salida
        self.W = np.random.random_sample((self.tamano_entrada, self.tamano_salida))
        self.colores_cat = cm.rainbow(np.linspace(0, 1, 10))

        #--------------------------------------------
        self.actualizarDataSet()

        self.M_sanger = np.tril(np.ones((self.tamano_salida,self.tamano_salida)))
        self.M_oja = np.ones((self.tamano_salida,self.tamano_salida))


    def actualizarDataSet(self):
        #--lectura de archivo de entrada
        data_input = np.genfromtxt(self.input_file, delimiter=',')
        #--preprocesamiento
        data_input[:,1:] = data_input[:,1:] / 10
        
        #--cambiamos el orden del dataset
        np.random.shuffle(data_input)
        #--tomamos el 75% para entrenar
        self.data_entrenamiento = data_input[0:len(data_input)*0.30,:]
        #--tomamos el 25% restante para validar
        self.data_validacion = data_input[len(data_input)*0.30 + 1:,:]

    def importarW(self, filename):
        csv_entrada = csv.reader(open(filename, "rb"), delimiter=',')
        
        cont = 0
        filas1 = 0
        for row in csv_entrada:
            if cont == 0:
                self.tamano_entrada = int(row[0])
                self.tamano_salida = int(row[1])
                self.W = np.zeros((self.tamano_entrada, self.tamano_salida))
            else:
                self.W[cont-1,:] = row
            cont += 1
        return "Red importada con exito!"

    def exportarW(self, filename):
        csv_salida = csv.writer(open(filename, "wb"))
        csv_salida.writerow([self.tamano_entrada, self.tamano_salida])
        for fila in self.W:
            csv_salida.writerow(fila)
        return "Red exportada con exito!"

    def cambiar_valor(self, clave, valor):
        if clave == 'tol':
            self.tolerancia_error = float(valor)
        elif clave == 'oja':
            self.oja = int(valor)
        elif clave == 'cep':
            self.cantidad_epocas = int(valor)
        elif clave == 'tin':
            tamano_viejo = self.tamano_entrada
            self.tamano_entrada = int(valor)
            if tamano_viejo < self.tamano_entrada:
                self.W = np.append(self.W, np.random.randn(self.tamano_entrada - tamano_viejo, self.tamano_salida), axis=1)
            else:
                self.W = self.W[0:self.tamano_entrada,:]
        elif clave == 'tout':
            tamano_viejo = self.tamano_salida
            self.tamano_salida = int(valor)
            if tamano_viejo < self.tamano_salida:
                self.W = np.append(self.W, np.random.randn(self.tamano_entrada, self.tamano_salida - tamano_viejo), axis=0)
            else:
                self.W = self.W[:,0:self.tamano_salida]
        elif clave == 'input':
            self.input_file = valor
            self.actualizarDataSet()
        elif clave == 'outtr':
            self.output_file_train = valor
        elif clave == 'outit':
            self.output_img_train = valor
        elif clave == 'outiv':
            self.output_img_valid = valor

    def entrenar(self):
        os.system("clear");
        print 'ENTRENAMIENTO:\n'
        
        #--se fija si tiene que haber salida
        csv_salida = 0
        if self.output_file_train != "":
            #--crea el archivo de salida y setea los labels (para el grafico)
            csv_salida = csv.writer(open(self.output_file_train, "wb"))
            csv_salida.writerow([self.input_file, self.tamano_salida, self.cantidad_epocas, self.tolerancia_error, self.oja])
            csv_salida.writerow(["Training: Convergencia de la ortonormalidad", "Error", "Epoca"])
            csv_salida.writerow(["Error"])


        x = np.zeros((1,self.tamano_entrada))
        y = np.zeros((1,self.tamano_salida))
        xp = np.zeros((self.tamano_salida, self.tamano_entrada))
        dW = np.zeros((self.tamano_entrada, self.tamano_salida))
        cant_repeticiones = 1
        error = 20
        num_epoca = 1

        epocas_igual = 0
        error_ultima_epoca = -1

        #--ciclo de entrenamiento por epocas
        while num_epoca <= self.cantidad_epocas and error > self.tolerancia_error and epocas_igual < 10:

            for fila in self.data_entrenamiento:

            	#-- distintas variantes de funcion de activacion
                fact_act = 1/(float(cant_repeticiones)**0.5)
                #fact_act = 1/(float(cant_repeticiones)**0.7)
                #fact_act = 1/float(cant_repeticiones)
                #fact_act = 0.01
                
                #-- toma los valores de entrada (descartando la categoría)
                x[:] = np.array([fila[1:self.tamano_entrada+1]])

                #-- calcula la salida multiplicando x con W
                y[:] = x.dot(self.W)

                if self.oja == 1:
                	#-- en oja usa la matriz M_oja, que tiene todos 1s,
                	# ya que oja hace la sumatoria de 1 a M
                	#-- luego multiplica por W traspuesta
                    xp[:] = np.multiply(self.M_oja, y).dot(self.W.T)
                else:
                	#-- en sanger usa la matriz M_sanger, que es triangular inferior
                	# ya que oja hace la sumatoria de 1 a J
                	#-- al ser triangular inferior, cuando toma la fila i tiene los
                	# valores y(1) hasta y(i-1)
                	#-- luego multiplica por W traspuesta
                    xp[:] = np.multiply(self.M_sanger, y).dot(self.W.T)

                #-- calcula la delta W haciendo la resta de x con xp y lo multiplica
                # por y
                dW[:] = fact_act * np.multiply( (x - xp).T, y)

                self.W[:] = self.W + dW
                cant_repeticiones += 1

            #-- toma el error -> diferencia cuadrática de la multiplicacion de W y
            # W traspuesta con la identidad (ortonormalidad)
            error = np.linalg.norm(self.W.T.dot(self.W) - np.identity(self.tamano_salida))

            #-- se fija si el error se mantiene igual
            if error == error_ultima_epoca:
                epocas_igual += 1
            else:
                epocas_igual = 0
                error_ultima_epoca = error

            
            #--imprime en el archivo de salida el error de cada epoca
            if csv_salida:
                csv_salida.writerow([num_epoca, error])

            
            print "Epoca "+str(num_epoca)+":"
            print "      Error: "+str(error)
            print ""

            num_epoca += 1
        
        csv_salida.writerow([])
        return "Fin entrenamiento"

    def graficar3dTrain(self):
        os.system("clear");

        plt.close('all')
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        for cat in xrange(1,10):
            pos_categoria = np.empty((0,self.dimension_final));
            for fila in self.data_entrenamiento:
                if int(fila[0]) == int(cat):
                    pos_categoria = np.append(pos_categoria, np.array([fila[1:]]).dot(self.W)[:,0:self.dimension_final], axis=0)
            if len(pos_categoria) > 0:
                ax.scatter(pos_categoria[:,0], pos_categoria[:,1], pos_categoria[:,2], c=self.colores_cat[cat], marker='o', edgecolors='none')

        plt.show()
        #plt.savefig(self.output_img_train)

        #--vista por cada parametro
        for i in xrange(0,self.dimension_final):
            plt.close('all')

            for cat in xrange(1,10):
                pos_categoria = np.empty((0,self.dimension_final));
                for fila in self.data_entrenamiento:
                    if int(fila[0]) == int(cat):
                        pos_categoria = np.append(pos_categoria, np.array([fila[1:]]).dot(self.W)[:,0:self.dimension_final], axis=0)
                if len(pos_categoria) > 0:
                    plt.scatter(pos_categoria[:,i], pos_categoria[:,(i+1) % self.dimension_final], c=self.colores_cat[cat], marker='o', edgecolors='none')

            #plt.show()
            plt.savefig("param_"+str(i)+"_"+self.output_img_train)
        
        raw_input("Pulse enter para volver al menu...")
        return "Fin grafico"


    def graficar3dValid(self):
        os.system("clear");

        plt.close('all')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for cat in xrange(1,10):
            pos_categoria = np.empty((0,3));
            for fila in self.data_validacion:
                if int(fila[0]) == int(cat):
                    pos_categoria = np.append(pos_categoria, np.array([fila[1:]]).dot(self.W), axis=0)
            if len(pos_categoria) > 0:
                ax.scatter(pos_categoria[:,0], pos_categoria[:,1], pos_categoria[:,2], zdir='z', c=self.colores_cat[cat], marker='o', edgecolors='none')

        #plt.show()
        plt.savefig(self.output_img_valid)

        raw_input("Pulse enter para volver al menu...")
        #plt.cla()
        return "Fin grafico"


def mostrar_menu(hebbian, msg):
    os.system('clear');
    print "MENU DEL EJ1:\n"
    print "Parametros activos:"
    print "  - tolerancia de error        (tol)        " + str(hebbian.tolerancia_error)
    print "  - cantidad de epocas         (cep)        " + str(hebbian.cantidad_epocas)
    print "  - oja (1->Oja; 0->Sanjer)    (oja)        " + str(hebbian.oja)
    print "  - tamano entrada             (tin)        " + str(hebbian.tamano_entrada)
    print "  - tamano salida              (tout)       " + str(hebbian.tamano_salida)
    print "  - input file                 (input)      " + str(hebbian.input_file)
    print "  - output file training       (outtr)      " + str(hebbian.output_file_train)
    print "  - output img training        (outit)      " + str(hebbian.output_img_train)
    print "  - output img validacion      (outiv)      " + str(hebbian.output_img_valid)
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
    print "AYUDA DEL EJ1:\n"
    print "Acciones validas:\n"
    print "  - Importar red neuronal"
    print "         Descripcion: importa una red neuronal desde un archivo"
    print "         Uso: import nombre_archivo.extension"
    print "         Ejemplo: import redOK.in"
    print ""
    print "  - Exportar red neuronal"
    print "         Descripcion: exporta la red neuronal actual hacia un archivo"
    print "         Uso: export nombre_archivo.extension"
    print "         Ejemplo: import red_v1.asd"
    print ""
    print "  - Cambiar parametro"
    print "         Descripcion: modifica el valor de un parametro"
    print "         Uso: change codigo_parametro nuevo_valor"
    print "         Ejemplo: change tol 0.001"
    print ""
    print "  - Comenzar entrenamiento"
    print "         Descripcion: ejecuta el entrenamiento con los parametros guardados"
    print "         Uso: train"
    print ""
    print "  - Graficar error de entrenamiento"
    print "         Descripcion: grafica lo que hay en el archivo de salida del entrenamiento"
    print "                      se debe haber ejecutado un entrenamiento antes."
    print "         Uso: graph train"
    print ""
    print "  - Graficar resultados de entrenamiento"
    print "         Descripcion: grafica la reduccion de los datos de entrenamiento a 3 dimensiones"
    print "         Uso: show train"
    print ""
    print "  - Graficar resultados de validacion"
    print "         Descripcion: grafica la reduccion de los datos de validacion a 3 dimensiones"
    print "         Uso: show valid"
    print ""

hebb = hebbian()

salir = 0
msg = ""

#hebb.importarW("red.in")

while (not salir):
    mostrar_menu(hebb, msg)
    msg = ""
    comando = raw_input("Ingrese un comando: ");
    comando = comando.split(" ")
    if (comando[0] == 'exit'):
        salir = 1
    elif (comando[0] == 'help'):
        mostrar_ayuda()
        raw_input("Pulse enter para volver al menu...")
    elif (comando[0] == 'train'):
        msg = hebb.entrenar()
    elif (comando[0] == 'change'):
        hebb.cambiar_valor(comando[1], comando[2])
    elif (comando[0] == 'export'):
        msg = hebb.exportarW(comando[1])
    elif (comando[0] == 'import'):
        msg = hebb.importarW(comando[1])
    elif comando[0] == 'graph' and comando[1] == 'train':
        msg = graficador.graficar_train(hebb.output_file_train)
    elif comando[0] == 'show':
        if comando[1] == 'color':
            hebb.graficarPorColor()
        elif comando[1] == 'train':
            hebb.graficar3dTrain()
        elif comando[1] == 'valid':
            hebb.graficar3dValid()