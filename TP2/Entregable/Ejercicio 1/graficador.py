import matplotlib.pyplot as plt
import csv
import sys
import numpy as np


def graficar_train(inputFile):
	#-----Formato esperado
	#	DATOS INSERVIBLES
	#	TITULO, LABEL Y, LABEL x
	#	LISTADO DE LEGENDS
	#	1, err1, err2, err3....
	#	2, err1, err2, err3....
	#	3, err1, err2, err3....
	#---------------------

	outputFile = inputFile

	entrada = csv.reader(open(inputFile, "rb"))

	linea = 0
	for row in entrada:
		if linea == 0:
			pass
		elif linea == 1:
			titulo = row[0]
			labely = row[1]
			labelx = row[2]
		elif linea == 2:
			legends = np.array(row)
			datos = np.empty((0,len(legends)+1))
			datos_zoom = np.empty((0,len(legends)+1))
		else:
			if len(row) > 0:
				datos = np.append(datos, np.array([row]), axis=0)
				if int(row[0]) >= 1000:
					datos_zoom = np.append(datos_zoom, np.array([row]), axis=0)

		linea = linea + 1


	print "GENERANDO GRAFICO..."
	for i in xrange(1,len(datos[0])):
		plt.plot(datos[:,0], datos[:,i])
	plt.legend(legends, loc='upper right')

	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.title(titulo)
	plt.legend()
	#plt.show()
	#raw_input()
	plt.savefig(outputFile+".png")

	plt.cla()

	print "GENERANDO GRAFICO ZOOM..."
	for i in xrange(1,len(datos_zoom[0])):
		plt.plot(datos_zoom[:,0], datos_zoom[:,i])
	plt.legend(legends, loc='upper right')

	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.title(titulo)
	plt.legend()
	#plt.show()
	#raw_input()
	plt.savefig(outputFile+"_zoom.png")

	return "grafico OK"