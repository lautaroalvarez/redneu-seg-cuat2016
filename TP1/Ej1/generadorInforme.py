#import graficador.py

def informe(cantidad_entradas,learning_rate,tolerancia_error, cantidad_repeticiones, cantidad_mezclas, input_file, output_file, tamano_capa, tamano_entrada, tamano_salida, beta1, beta2):

	print "Generando Informe..."

	f = open('informe.html','w')
	informe = """
<html>
	<link href="css/estilos.css" rel="stylesheet" type="text/css">
	<head>
		<title>Resumen</title>
		<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
	</head>

	<body bgcolor="#FFFFFF" text="#000000">
		<table width="100%" border="0" cellpadding="0" cellspacing="0">
			<tr> 
				<td width="120" height="40" align="top" bgcolor="#00005A"></td>
				<td class="titulo-general"> 
					Trabajo Practico 1 - Ejercicio 1 - Resumen de Ejecucion
				</td>
				<td width="120" height="40" align="top"  bgcolor="#00005A"></td>
			</tr>
		</table>
	
		<table><tr><br></tr></table>


		<div class="tex-azul-Arial-bold-11" >
	  		<p class="ItemMenu"> Datos Ingresados por Parametro </p>
		</div>


		<table style="border: 1 #CDCAC9 solid" width="50%" Cellspacing="5" Cellpadding="5">
			<tr>
				<td width="10%" Class="tex-negro-Arial-11">Archivo Entrada: </td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{input_file}
				</td>
			</tr>
			<tr>
				<td width="10%" Class="tex-negro-Arial-11">Archivo Salida: </td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{output_file}
				</td>
			</tr>

			<tr>
				<td width="10%" Class="tex-negro-Arial-11">Cantidad de datos tomados: </td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{cantidad_entradas}
				</td>
			</tr>
			<tr>
				<td width="10%" Class="tex-negro-Arial-11">Learning Rate: </td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{learning_rate}
				</td>
			</tr>
			<tr>		
				<td width="10%" Class="tex-negro-Arial-11">Tolerancia de Error:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{tolerancia_error}
				</td>
			</tr>
			<tr>		
			     	<td width="10%" Class="tex-negro-Arial-11">Cantidad de Repeticiones:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{cantidad_repeticiones}
				</td>
			</tr>
			<tr>							
			    	<td width="10%" Class="tex-negro-Arial-11">Cantidad de Mezclas:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{cantidad_mezclas}
				</td>
			</tr>
			<tr>		
				<td width="10%" Class="tex-negro-Arial-11">Tamano Entrada:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{tamano_entrada}
				</td>
			</tr>
			<tr>		
				<td width="10%" Class="tex-negro-Arial-11">Tamano Salida:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{tamano_salida}
				</td>
			</tr>
			<tr>		
				<td width="10%" Class="tex-negro-Arial-11">Tamano Capa:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{tamano_capa}
				</td>						
			</tr>
			<tr>		
				<td width="10%" Class="tex-negro-Arial-11">Beta Activacion 1 Capa:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{beta1}
				</td>						
			</tr>
			<tr>		"ME%d" % i
				<td width="10%" Class="tex-negro-Arial-11">Beta Activacion 2 Capa:</td>
				<td width="10%" Class="tex-azul-Arial-bold-11" >
					{beta2}
				</td>						
			</tr>
		</table>

		<div class="tex-azul-Arial-bold-11" >
  			<p class="ItemMenu"> Resultados Obtenidos para cada Etapa </p>
		</div>""".format(input_file=input_file, output_file=output_file, cantidad_entradas=cantidad_entradas, learning_rate=learning_rate, tolerancia_error=tolerancia_error, cantidad_repeticiones=cantidad_repeticiones, cantidad_mezclas=cantidad_mezclas, tamano_entrada=tamano_entrada, tamano_salida=tamano_salida, tamano_capa=tamano_capa, beta1=beta1, beta2=beta2)

	for etapa in xrange(0, cantidad_mezclas):
		imagen_ecm = "mezcla10_ecm_" + str(etapa) + ".jpg"
		imagen_promedios ="mezcla10_" + str(etapa) + ".jpg"
		informe = informe + """<div>
			
			<p class="tex-negro-Arial-12">Etapa: {etapa}</p>
			<img src="{imagen_ecm}" alt="Convergencia del Error">			
			<img src="{imagen_promedios}" alt="Error Cuadratico Medio">
			
	  		<p class="tex-negro-Arial-11"> Error Cuadratico Medio Promedio: aca va el ECM</p>
			<p class="tex-negro-Arial-11"> Testing de la Etapa: aca va el testing de la etapa</p>
		</div>""".format(etapa=etapa, imagen_ecm=imagen_ecm, imagen_promedios=imagen_promedios)

	informe = informe + """<div class="tex-azul-Arial-bold-11" >
  			<p class="ItemMenu"> Testing Final </p>
		</div>

		<div>
  			<p class="tex-negro-Arial-12"> Cantidad de Aciertos: aca va la cantidad de aciertos del cross validation</p>

			<p class="tex-negro-Arial-12"> Total Casos: aca va el total de los casos</p>

			<p class="tex-negro-Arial-12"> Efectividad (%): el porcentaje de efectividad </p>
		</div>


	</body>
</html>"""

	f.write(informe)
	f.close()
	print "Informe Finalizado."
	return


def graficos (inputFile, outputFile, cantCol):
	columnas = []
	for i in xrange(3,len(sys.argv)):
		columnas.append(sys.argv[i])

	entrada = csv.reader(open(inputFile, "rb"))

	cant_mezlcas = 0
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
		else:
			if len(row) == 0:
				print "Generando Graficos..."
				if len(columnas)==0:
					for i in xrange(1,len(datos[0])):
						plt.plot(datos[:,0], datos[:,i])
					plt.legend(legends, loc='upper right')
				else:
					legends_reducidas = []
					for i in xrange(0, len(columnas)):
						plt.plot(datos[:,0], datos[:,columnas[i]])
						legends_reducidas.append(legends[(int)(columnas[i])-1])
					plt.legend(legends_reducidas, loc='upper right')
	
				plt.xlabel(labelx)
				plt.ylabel(labely)
				plt.title(titulo)
				plt.legend()
				plt.savefig(outputFile+"_"+str(cant_mezlcas)+".png")
				cant_mezlcas = cant_mezlcas + 1
				datos = np.empty((0,len(legends)+1))
				plt.gcf().clear()
			else:
				datos = np.append(datos, np.array([row]), axis=0)
	
		linea = linea + 1
	print "Graficos Generados"
	return;
