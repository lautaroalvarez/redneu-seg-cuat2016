import csv, sys, math, os
import numpy as np
import graficador


def getDataSet(filename):
    datos = np.genfromtxt(filename, delimiter=',')
    return datos

def preprocesamiento(datos):
    for i in xrange(0,len(datos[0])-2):
        datos[:,i] = (datos[:,i] - np.median(datos[:,i])) / np.std(datos[:,i])

    return

class perceptron:
    def __init__(self):
        #--------------------------------------------
        #------Parametros
        self.learning_rate = 0.1
        self.beta1 = 0.1 #--beta en la funcion de activacion 1
        self.beta2 = 0.1 #--beta en la funcion de activacion 2
        self.tolerancia_error = 1
        self.cantidad_repeticiones = 200
        self.cantidad_mezclas = 5

        self.input_file = "tp1_ej2_training.csv" #--archivo csv de entrada
        self.output_file_train = "ej2_train.sal" #--archivo csv de salida de training
        self.output_file_valid = "ej2_valid.sal" #--archivo csv de salida de validacion
        self.output_file_test1 = "test1.sal" #--archivo csv de salida del test
        self.output_file_test2 = "test2.sal" #--archivo csv de salida del test

        self.tamano_capa = 6
        self.tamano_entrada = 8 #--dimension de la entrada
        self.tamano_salida = 2 #--dimension de la salida
        #--------------------------------------------

        self.w1 = np.random.randn(self.tamano_entrada+1, self.tamano_capa)
        self.w2 = np.random.randn(self.tamano_capa, self.tamano_salida)

    def importarW(self, filename):
        csv_entrada = csv.reader(open(filename, "rb"), delimiter=',')
        
        cont = 0
        filas1 = 0
        columnas1 = 0
        filas2 = 0
        columnas2 = 0
        for row in csv_entrada:
            if cont == 0:
                filas1 = row[0]
                columnas1 = row[1]
                filas2 = row[2]
                columnas2 = row[3]
                self.tamano_capa = int(columnas1)
                self.tamano_entrada = int(filas1)-1
                self.tamano_salida = int(columnas2)
                self.w1 = np.zeros((self.tamano_entrada+1, self.tamano_capa))
                self.w2 = np.zeros((self.tamano_capa, self.tamano_salida))
            elif cont <= int(filas1):
                self.w1[cont-1,:] = row
            elif cont < cont+int(filas2):
                self.w2[cont-int(filas1)-1,:] = row
            cont = cont + 1
        return "Red importada con exito!"

    def exportarW(self, filename):
        csv_salida = csv.writer(open(filename, "wb"))
        csv_salida.writerow([self.w1.shape[0], self.w1.shape[1], self.w2.shape[0], self.w2.shape[1]])
        for i in xrange(0,len(self.w1)):
            csv_salida.writerow(self.w1[i])
        for i in xrange(0,len(self.w2)):
            csv_salida.writerow(self.w2[i])
        return "Red exportada con exito!"

    def funcion_activacion_1(self, m):
        return np.tanh(m)

    def funcion_activacion_2(self, m):
        return abs(m)

    def funcion_activacion_derivada_1(self, m):
        return 1.0 - np.tanh(m)**2

    def funcion_activacion_derivada_2(self, m):
        return np.ones(m.shape)

    def cambiar_valor(self, clave, valor):
        if clave == 'lr':
            self.learning_rate = float(valor)
        elif clave == 'tol':
            self.tolerancia_error = float(valor)
        elif clave == 'crep':
            self.cantidad_repeticiones = float(valor)
        elif clave == 'tin':
            self.tamano_entrada = float(valor)
        elif clave == 'tout':
            self.tamano_salida = float(valor)
        elif clave == 'tcapa':
            tamano_viejo = self.tamano_capa
            self.tamano_capa = float(valor)
            if tamano_viejo < self.tamano_capa:
                self.w1 = np.append(self.w1, np.random.randn(self.tamano_entrada+1, self.tamano_capa - tamano_viejo), axis=1)
                self.w2 = np.append(self.w2, np.random.randn(self.tamano_capa - tamano_viejo, self.tamano_salida), axis=0)
            else:
                self.w1 = self.w1[:,0:self.tamano_capa]
                self.w2 = self.w2[0:self.tamano_capa,:]
        elif clave == 'input':
            self.input_file = valor
        elif clave == 'outtr':
            self.output_file_train = valor
        elif clave == 'outva':
            self.output_file_valid = valor
        elif clave == 'outt1':
            self.output_file_test1 = valor
        elif clave == 'outt2':
            self.output_file_test2 = valor
        elif clave == 'cmez':
            self.cantidad_mezclas = int(valor)

    def entrenar(self):
        os.system("clear");
        print "INICIANDO ENTRENAMIENTO"
        
        #--busca datos del archivo de entrada
        data_input = getDataSet(perc.input_file)
        preprocesamiento(data_input)

        #--se fija si tiene que haber salida
        csv_salida = 0
        if self.output_file_train != "":
            #--crea el archivo de salida y setea los labels (para el grafico)
            csv_salida = csv.writer(open(self.output_file_train, "wb"))
            csv_salida.writerow([self.input_file, self.learning_rate, self.cantidad_repeticiones, self.tolerancia_error, self.beta1, self.beta2])
            csv_salida.writerow(["Training: convergencia del error", "Error", "Etapa"])
            csv_salida.writerow(["Error promedio", "Error minimo", "Error maximo"])

        csv_salida2 = 0
        if self.output_file_valid != "":
            #--crea el archivo de salida y setea los labels (para el grafico)
            csv_salida2 = csv.writer(open(self.output_file_valid, "wb"))
            csv_salida2.writerow([self.input_file, self.learning_rate, self.cantidad_repeticiones, self.tolerancia_error, self.beta1, self.beta2])
            csv_salida2.writerow(["Validacion: convergencia del error", "Error", "Etapa"])
            csv_salida2.writerow(["Error promedio", "Error minimo", "Error maximo"])

        resultados_muestra = np.zeros(self.cantidad_mezclas)
        resultados_verificacion = np.zeros(self.cantidad_mezclas)

        for k in xrange(0, self.cantidad_mezclas):

            np.random.shuffle(data_input)

            input_x = data_input[:,0:8]
            input_y = data_input[:,8:10]

            cant_repeticiones = 0
            error_promedio = 20
            cantidad_entradas = int(len(input_x)*0.8)
            #--ciclo de entrenamiento por etapas
            while cant_repeticiones < self.cantidad_repeticiones and error_promedio > self.tolerancia_error:
            
                error_acumulado = 0
                error_promedio = 0
                error_minimo = 2000000
                error_maximo = 0

                dw2 = self.w2
                dw1 = self.w1

                for i in xrange(0, cantidad_entradas):
                    #--setea la salida esperada
                    salida_esperada = np.array([input_y[i]]).T

                    #--setea los distintos arrays y matrices necesarios en cero
                    p = np.zeros((self.tamano_entrada+1, 1))    #--input
                    n1 = np.zeros((self.tamano_capa, 1))   #--resultado de p*w1
                    a1 = np.zeros((self.tamano_capa, 1))   #--resultado de aplicarle la funcion de activacion 1 a n1
                    n2 = np.zeros((self.tamano_salida, 1)) #--resultado de n1*w2
                    a2 = np.zeros((self.tamano_salida, 1)) #--resultado de aplicarle la funcion de activacion 1 a n2

                    #--setea el input 'p' segun la fila actual del archivo de entrada
                    p[0:self.tamano_entrada] = np.array([input_x[i]]).T
                    #--agrega baeas
                    p[self.tamano_entrada][0] = -1

                    #--activacion
                    n1[:] = self.w1.T.dot(p)
                    a1[:] = self.funcion_activacion_1( n1 )
                    n2[:] = self.w2.T.dot(a1)
                    a2[:] = self.funcion_activacion_2( n2 )

                    #--correccion
                    d2 = salida_esperada - a2
                    error = abs(d2)
                    d1 = self.w2.dot(d2)

                    #--Acumula los delta w
                    dw2[:] = dw2 + self.learning_rate * np.dot(a1, np.multiply( self.funcion_activacion_derivada_2( n2.T ), d2.T ) )
                    dw1[:] = dw1 + self.learning_rate * np.dot(p, np.multiply( self.funcion_activacion_derivada_1( n1.T ), d1.T ) )

                    #--acumula el error
                    sum_error = 0
                    for i in xrange(0, self.tamano_salida):
                        sum_error = sum_error + (error[i][0] ** 2)
                    error_cuad = math.sqrt(sum_error)

                    error_acumulado = error_acumulado + error_cuad
                    
                    #--se fija si el error es maximo
                    if error_cuad > error_maximo:
                        error_maximo = error_cuad
                    #--se fija si el error es minimo
                    if error_cuad < error_minimo:
                        error_minimo = error_cuad

                #--calcula el error promedio
                error_promedio = error_acumulado / cantidad_entradas

                #--apaptacion
                self.w2[:] = dw2
                self.w1[:] = dw1

                #--imprime en el archivo de salida
                if csv_salida:
                    csv_salida.writerow([cant_repeticiones, error_promedio, error_minimo, error_maximo])
                
                resultados_muestra[k] = error_promedio
                resultados_verificacion[k]  = self.validacion(data_input[cantidad_entradas:len(data_input)], cant_repeticiones, csv_salida2)
                
                
                os.system('clear');
                print 'ENTRENAMIENTO:\n'
                for j in xrange(0,k+1):
                    print "Etapa "+str(j)+":"
                    print "      Error Promedio: "+str(resultados_muestra[j])
                    if j!=k:
                        print "      Testing: "+str(resultados_verificacion[j])+"/"+str(len(input_x))
                    print ""

                cant_repeticiones = cant_repeticiones + 1
            
            csv_salida.writerow([])
            csv_salida2.writerow([])

        return "Fin entrenamiento"


    def validacion(self, data_input, num, csv_salida2):
        input_y = data_input[:,8:10]
        input_x = data_input[:,0:8]


        cantidad_ok = 0
        error_acumulado = 0
        error_maximo = 0
        error_minimo = 20000000
        for i in xrange(0, len(input_x)):
            #--setea la salida esperada
            salida_esperada = np.array([input_y[i]]).T

            #--setea los distintos arrays y matrices necesarios en cero
            p = np.zeros((self.tamano_entrada+1, 1))    #--input
            n1 = np.zeros((self.tamano_capa, 1))   #--resultado de p*w1
            a1 = np.zeros((self.tamano_capa, 1))   #--resultado de aplicarle la funcion de activacion 1 a n1
            n2 = np.zeros((self.tamano_salida, 1)) #--resultado de n1*w2
            a2 = np.zeros((self.tamano_salida, 1)) #--resultado de aplicarle la funcion de activacion 1 a n2

            #--setea el input 'p' segun la fila actual del archivo de entrada
            p[0:self.tamano_entrada] = np.array([input_x[i]]).T
            #--agrega baeas
            p[self.tamano_entrada][0] = -1

            #--activacion
            n1[:] = self.w1.T.dot(p)
            a1[:] = self.funcion_activacion_1( n1 )
            n2[:] = self.w2.T.dot(a1)
            a2[:] = self.funcion_activacion_2( n2 )


            #--determinar salida
            sum_error = 0
            dif = salida_esperada - a2
            for j in xrange(0, self.tamano_salida):
                sum_error = sum_error + (dif[j][0] ** 2)
            error_cuad = math.sqrt(sum_error)
            if error_cuad < self.tolerancia_error:
                cantidad_ok = cantidad_ok + 1

            error_acumulado = error_acumulado + error_cuad

            if error_cuad > error_maximo:
                error_maximo = error_cuad
            if error_cuad < error_minimo:
                error_minimo = error_cuad

        error_promedio = error_acumulado / len(input_x)

        #--imprime en el archivo de salida
        if csv_salida2:
            csv_salida2.writerow([num, error_promedio, error_minimo, error_maximo])

        return cantidad_ok

    def testing(self):
        #--busca datos del archivo de entrada
        data_input = getDataSet(perc.input_file)
        input_y = data_input[:,8:10]
        #preprocesamiento(data_input[:,0:8])
        input_x = data_input[:,0:8]


        #----IMPRIMO ARCHIVO DE SALIDAS VS ERROR
        csv_salida = 0
        if self.output_file_test1 != "":
            csv_salida = csv.writer(open(self.output_file_test1, "wb"))
            csv_salida.writerow([])
            csv_salida.writerow(["Calefaccion: salida esperada vs salida de la red", "Error", "Instancia"])
            csv_salida.writerow(["Salida esperada", "Salida de la red"])

        csv_salida2 = 0
        if self.output_file_test2 != "":
            csv_salida2 = csv.writer(open(self.output_file_test2, "wb"))
            csv_salida2.writerow([])
            csv_salida2.writerow(["Refrigeracion: salida esperada vs salida de la red", "Error", "Instancia"])
            csv_salida2.writerow(["Salida esperada", "Salida de la red"])
        

        cantidad_ok = 0
        for i in xrange(0, len(input_x)):
            #--setea la salida esperada
            salida_esperada = np.array([input_y[i]]).T

            #--setea los distintos arrays y matrices necesarios en cero
            p = np.zeros((self.tamano_entrada+1, 1))    #--input
            n1 = np.zeros((self.tamano_capa, 1))   #--resultado de p*w1
            a1 = np.zeros((self.tamano_capa, 1))   #--resultado de aplicarle la funcion de activacion 1 a n1
            n2 = np.zeros((self.tamano_salida, 1)) #--resultado de n1*w2
            a2 = np.zeros((self.tamano_salida, 1)) #--resultado de aplicarle la funcion de activacion 1 a n2

            #--setea el input 'p' segun la fila actual del archivo de entrada
            p[0:self.tamano_entrada] = np.array([input_x[i]]).T
            #--agrega baeas
            p[self.tamano_entrada][0] = -1

            #--activacion
            n1[:] = self.w1.T.dot(p)
            a1[:] = self.funcion_activacion_1( n1 )
            n2[:] = self.w2.T.dot(a1)
            a2[:] = self.funcion_activacion_2( n2 )


            #--determinar salida
            sum_error = 0
            dif = salida_esperada - a2
            for j in xrange(0, self.tamano_salida):
                sum_error = sum_error + (dif[j][0] ** 2)
            error_cuad = math.sqrt(sum_error)
            if error_cuad < self.tolerancia_error:
                cantidad_ok = cantidad_ok + 1

            if csv_salida:
                csv_salida.writerow([i, salida_esperada[0][0], a2[0][0]])
            if csv_salida2:
                csv_salida2.writerow([i, salida_esperada[1][0], a2[1][0]])

        print "RESULTADO DEL TESTING"
        print str(cantidad_ok)+"/"+str(len(input_x))+" correctos"
        return cantidad_ok

def mostrar_menu(perc, msg):
    os.system('clear');
    print "MENU DEL EJ1:\n"
    print "Parametros activos:"
    print "  - learning rate              (lr)         " + str(perc.learning_rate)
    print "  - tolerancia de error        (tol)        " + str(perc.tolerancia_error)
    print "  - cantidad repeticiones      (crep)       " + str(perc.cantidad_repeticiones)
    print "  - cantidad mezclas           (cmez)       " + str(perc.cantidad_mezclas)
    print "  - tamano entrada             (tin)        " + str(perc.tamano_entrada)
    print "  - tamano salida              (tout)       " + str(perc.tamano_salida)
    print "  - tamano capa oculta         (tcapa)      " + str(perc.tamano_capa)
    print "  - input file                 (input)      " + str(perc.input_file)
    print "  - output file training       (outtr)      " + str(perc.output_file_train)
    print "  - output file validacion     (outva)      " + str(perc.output_file_valid)
    print "  - out file del test (campo 1)(outt1)      " + str(perc.output_file_test1)
    print "  - out file del test (campo 2)(outt2)      " + str(perc.output_file_test2)
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
    print "  - Graficar entrenamiento"
    print "         Descripcion: grafica lo que hay en el archivo de salida del entrenamiento"
    print "                      se deberia haber ejecutado un entrenamiento antes."
    print "         Uso: graph train"
    print ""
    print "  - Graficar validacion"
    print "         Descripcion: grafica lo que hay en el archivo de salida de la validacion del entrenamiento"
    print "                      se deberia haber ejecutado un entrenamiento antes."
    print "         Uso: graph valid"
    print ""
    print "  - Graficar testing"
    print "         Descripcion: grafica lo que hay en los archivos de salida del testing"
    print "                      se deberia haber ejecutado un testing antes."
    print "         Uso: graph test"
    print ""

perc = perceptron()

salir = 0
msg = ""

#Agregado para informe
cantidad_entradas =0


while (not salir):
    mostrar_menu(perc, msg)
    msg = ""
    comando = raw_input("Ingrese un comando: ");
    comando = comando.split(" ")
    if (comando[0] == 'exit'):
        salir = 1
    elif (comando[0] == 'help'):
        mostrar_ayuda()
        raw_input("Pulse enter para volver al menu...")
    elif (comando[0] == 'train'):
        msg = perc.entrenar()
        raw_input("Pulse enter para volver al menu...")
    elif (comando[0] == 'change'):
        perc.cambiar_valor(comando[1], comando[2])
    elif (comando[0] == 'export'):
        msg = perc.exportarW(comando[1])
    elif (comando[0] == 'import'):
        msg = perc.importarW(comando[1])
    elif (comando[0] == 'test'):
        perc.testing()
        raw_input("Pulse enter para volver al menu...")
    elif (comando[0] == 'reporte'):
        perc.reporte()
        raw_input("Pulse enter para volver al menu...")
    elif comando[0] == 'graph':
        if comando[1] == 'train':
            msg = graficador.graficar_train(perc.output_file_train)
        if comando[1] == 'valid':
            msg = graficador.graficar_train(perc.output_file_valid)
        elif comando[1] == 'test':
            graficador.graficar_test(perc.output_file_test1)
            msg = graficador.graficar_test(perc.output_file_test2)
