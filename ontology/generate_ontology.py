#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script general para generar ontología OWL basada en archivos CSV
"""

import pandas as pd
import re
import os
import glob
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
from rdflib.namespace import XSD


def to_pascal_case(text):
    """Convierte texto a PascalCase"""
    # Separar por guiones bajos y espacios
    words = re.split(r'[_\s]+', text.lower())
    # Capitalizar primera letra de cada palabra
    return ''.join(word.capitalize() for word in words if word)


def to_camel_case(text):
    """Convierte texto a camelCase (primera letra minúscula)"""
    pascal = to_pascal_case(text)
    if pascal:
        return pascal[0].lower() + pascal[1:]
    return pascal


def generate_ontology_from_csv(csv_file_path, output_file_path, ontology_name=None):
    """
    Genera una ontología OWL basada en la información de un archivo CSV
    
    Args:
        csv_file_path (str): Ruta al archivo CSV
        output_file_path (str): Ruta donde guardar la ontología
        ontology_name (str): Nombre personalizado para la ontología (opcional)
    """
    
    # Leer el CSV
    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV leído exitosamente. {len(df)} filas encontradas.")
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return
    
    # Verificar que las columnas necesarias existan
    required_columns = ['atribute', 'table', 'property', 'range', 'dictionary_group', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Faltan las siguientes columnas en el CSV: {missing_columns}")
        return
    
    # Generar nombre de ontología basado en el archivo CSV si no se proporciona
    if ontology_name is None:
        csv_filename = os.path.basename(csv_file_path)
        ontology_name = os.path.splitext(csv_filename)[0]
    
    csv_filename = os.path.basename(csv_file_path)
    
    # Configurar namespace
    base_uri = f"http://infmed.fcrb.es/ontologias/{ontology_name.lower()}"
    namespace_uri = f"{base_uri}#"
    
    # Configurar namespace para el diccionario
    dict_base_uri = f"http://infmed.fcrb.es/ontologias/{ontology_name.lower()}_dic"
    dict_namespace_uri = f"{dict_base_uri}#"
    
    # Definir namespaces
    ns = Namespace(namespace_uri)
    dict_ns = Namespace(dict_namespace_uri)
    
    # Crear el grafo RDF para la ontología principal
    g = Graph()
    
    # Crear el grafo RDF para el diccionario
    dict_g = Graph()
    
    # Configurar namespaces con bind antes de crear elementos
    g.bind(ontology_name.lower(), ns)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind(f"{ontology_name.lower()}_dic", dict_ns)
    
    dict_g.bind(f"{ontology_name.lower()}_dic", dict_ns)
    dict_g.bind("owl", OWL)
    dict_g.bind("rdfs", RDFS)
    dict_g.bind("xsd", XSD)
    
    # Crear la ontología principal
    ontology_uri = URIRef(base_uri)
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal(f"{ontology_name} Ontology")))
    g.add((ontology_uri, RDFS.comment, Literal(f"Ontología generada automáticamente basada en {csv_filename}")))
    
    # Importar la ontología de diccionario
    g.add((ontology_uri, OWL.imports, URIRef(dict_base_uri)))
    
    # Crear la ontología de diccionario
    dict_ontology_uri = URIRef(dict_base_uri)
    dict_g.add((dict_ontology_uri, RDF.type, OWL.Ontology))
    dict_g.add((dict_ontology_uri, RDFS.label, Literal(f"{ontology_name} Dictionary Ontology")))
    dict_g.add((dict_ontology_uri, RDFS.comment, Literal(f"Diccionario de clases para la ontología {ontology_name}")))
    
    # Crear la clase principal Dictionary en el diccionario
    dictionary_class = dict_ns.Dictionary
    dict_g.add((dictionary_class, RDF.type, OWL.Class))
    dict_g.add((dictionary_class, RDFS.label, Literal("Dictionary")))
    dict_g.add((dictionary_class, RDFS.comment, Literal("Clase principal del diccionario que contiene todas las clases de referencia")))
    
    # Crear la clase InstanceIdentifier al mismo nivel que Dictionary
    instance_identifier_class = dict_ns.InstanceIdentifier
    dict_g.add((instance_identifier_class, RDF.type, OWL.Class))
    dict_g.add((instance_identifier_class, RDFS.label, Literal("InstanceIdentifier")))
    dict_g.add((instance_identifier_class, RDFS.comment, Literal("Clase para identificadores de instancia")))
    
    # Crear propiedades datatype para la clase Dictionary
    has_code_prop = dict_ns.hasCode
    dict_g.add((has_code_prop, RDF.type, OWL.DatatypeProperty))
    dict_g.add((has_code_prop, RDF.type, OWL.FunctionalProperty))
    dict_g.add((has_code_prop, RDFS.label, Literal("has code", lang="en")))
    dict_g.add((has_code_prop, RDFS.comment, Literal("Código identificador del elemento del diccionario")))
    dict_g.add((has_code_prop, RDFS.domain, dictionary_class))
    dict_g.add((has_code_prop, RDFS.range, XSD.string))

    has_domain_prop = dict_ns.hasDomain
    dict_g.add((has_domain_prop, RDF.type, OWL.DatatypeProperty))
    dict_g.add((has_domain_prop, RDF.type, OWL.FunctionalProperty))
    dict_g.add((has_domain_prop, RDFS.label, Literal("has domain", lang="en")))
    dict_g.add((has_domain_prop, RDFS.comment, Literal("Dominio del elemento del diccionario")))
    dict_g.add((has_domain_prop, RDFS.domain, dictionary_class))
    dict_g.add((has_domain_prop, RDFS.range, XSD.string))
    
    has_description_prop = dict_ns.hasDescription
    dict_g.add((has_description_prop, RDF.type, OWL.DatatypeProperty))
    dict_g.add((has_description_prop, RDF.type, OWL.FunctionalProperty))
    dict_g.add((has_description_prop, RDFS.label, Literal("has description", lang="en")))
    dict_g.add((has_description_prop, RDFS.comment, Literal("Descripción del elemento del diccionario")))
    dict_g.add((has_description_prop, RDFS.domain, dictionary_class))
    dict_g.add((has_description_prop, RDFS.range, XSD.string))
    
    # Crear propiedad especial para InstanceIdentifier
    # has_ii_value_prop = dict_ns.hasIIValue
    # dict_g.add((has_ii_value_prop, RDF.type, OWL.DatatypeProperty))
    # dict_g.add((has_ii_value_prop, RDF.type, OWL.FunctionalProperty))
    # dict_g.add((has_ii_value_prop, RDFS.label, Literal("has instance identifier value", lang="en")))
    # dict_g.add((has_ii_value_prop, RDFS.comment, Literal("Valor del identificador de instancia")))
    # dict_g.add((has_ii_value_prop, RDFS.domain, instance_identifier_class))
    # dict_g.add((has_ii_value_prop, RDFS.range, XSD.string))
    
    # Crear la clase principal "Tables" usando el namespace correcto
    tables_class = ns.Tables
    g.add((tables_class, RDF.type, OWL.Class))
    g.add((tables_class, RDFS.label, Literal("Tables")))
    g.add((tables_class, RDFS.comment, Literal("Clase principal que contiene todas las tablas del modelo")))
    
    # Obtener tablas únicas
    unique_tables = df['table'].unique()
    table_classes = {}
    
    # Crear clases para cada tabla usando fragmento #
    for table in unique_tables:
        # Convertir nombre de tabla a PascalCase
        class_name = to_pascal_case(table)
        class_uri = ns[class_name]
        
        # Crear la clase
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(class_name)))
        g.add((class_uri, RDFS.comment, Literal(f"Clase para la tabla {table}")))
        
        # Hacer que sea subclase de Tables
        g.add((class_uri, RDFS.subClassOf, tables_class))
        
        table_classes[table] = class_uri
        print(f"Clase creada: {class_name} para tabla {table}")
    
    # Crear propiedades para cada atributo
    properties_created = set()
    range_classes_created = set()
    dictionary_group_classes = set()
    
    for _, row in df.iterrows():
        attribute = row['atribute']
        table = row['table']
        property_type = row['property']
        range_value = row['range']
        dictionary_group = row['dictionary_group'] if pd.notna(row['dictionary_group']) else None
        label_value = row['label']
        
        # Generar nombre de propiedad con nomenclatura "has" + Clase + Atributo usando fragmento #
        table_class_name = to_pascal_case(table)
        attribute_name = to_pascal_case(attribute)
        property_name = "has" + table_class_name + attribute_name
        property_uri = ns[property_name]
        
        # Evitar duplicados
        if property_name in properties_created:
            continue
        
        properties_created.add(property_name)
        
        # Determinar el tipo de propiedad basado en la columna 'property'
        if property_type == 'O':
            # Object Property
            g.add((property_uri, RDF.type, OWL.ObjectProperty))
            g.add((property_uri, RDF.type, OWL.FunctionalProperty))
            property_type_label = "Object Property"
            
            # Si hay dictionary_group, crear la clase agrupadora
            if dictionary_group:
                # Crear clase de grupo en el diccionario
                group_class_name = to_pascal_case(dictionary_group)
                group_class_uri = dict_ns[group_class_name]
                
                # Tratamiento especial para InstanceIdentifier - ya está creada
                if group_class_name == "InstanceIdentifier":
                    group_class_uri = instance_identifier_class
                    if group_class_name not in dictionary_group_classes:
                        dictionary_group_classes.add(group_class_name)
                        print(f"Usando clase InstanceIdentifier ya creada")
                else:
                    # Para otras clases de grupo, crearlas como subclases de Dictionary
                    if group_class_name not in dictionary_group_classes:
                        dict_g.add((group_class_uri, RDF.type, OWL.Class))
                        dict_g.add((group_class_uri, RDFS.label, Literal(group_class_name)))
                        dict_g.add((group_class_uri, RDFS.comment, Literal(f"Clase de grupo para {dictionary_group}")))
                        dict_g.add((group_class_uri, RDFS.subClassOf, dictionary_class))
                        dictionary_group_classes.add(group_class_name)
                        print(f"Clase de grupo creada en diccionario: {group_class_name}")
                
                # Crear clase específica en el diccionario basada en el valor de range en PascalCase
                range_class_name = to_pascal_case(range_value)
                range_class_uri = dict_ns[range_class_name]
                
                # Crear la clase solo si no existe
                if range_class_name not in range_classes_created:
                    dict_g.add((range_class_uri, RDF.type, OWL.Class))
                    dict_g.add((range_class_uri, RDFS.label, Literal(range_class_name)))
                    dict_g.add((range_class_uri, RDFS.comment, Literal(f"Clase para {range_value}")))
                    dict_g.add((range_class_uri, RDFS.subClassOf, group_class_uri))
                    range_classes_created.add(range_class_name)
                    print(f"Clase de rango creada en diccionario: {range_class_name} bajo grupo {group_class_name}")
            else:
                # Sin dictionary_group, crear clase directamente bajo Dictionary
                range_class_name = to_pascal_case(range_value)
                range_class_uri = dict_ns[range_class_name]
                
                if range_class_name not in range_classes_created:
                    dict_g.add((range_class_uri, RDF.type, OWL.Class))
                    dict_g.add((range_class_uri, RDFS.label, Literal(range_class_name)))
                    dict_g.add((range_class_uri, RDFS.comment, Literal(f"Clase para {range_value}")))
                    dict_g.add((range_class_uri, RDFS.subClassOf, dictionary_class))
                    range_classes_created.add(range_class_name)
                    print(f"Clase de rango creada en diccionario: {range_class_name}")
            
            # Establecer rango de la object property
            g.add((property_uri, RDFS.range, range_class_uri))
            
        elif property_type == 'D':
            # Datatype Property
            g.add((property_uri, RDF.type, OWL.DatatypeProperty))
            g.add((property_uri, RDF.type, OWL.FunctionalProperty))
            property_type_label = "Datatype Property"
            
            # Establecer rango basado en el valor de range
            if range_value == 'datetime':
                g.add((property_uri, RDFS.range, XSD.dateTime))
            elif range_value == 'float':
                g.add((property_uri, RDFS.range, XSD.float))
            elif range_value == 'int':
                g.add((property_uri, RDFS.range, XSD.integer))
            elif range_value == 'string' or pd.isna(range_value):
                g.add((property_uri, RDFS.range, XSD.string))
            else:
                # Por defecto string si no se reconoce el tipo
                g.add((property_uri, RDFS.range, XSD.string))
        else:
            # Por defecto, Object Property
            g.add((property_uri, RDF.type, OWL.ObjectProperty))
            g.add((property_uri, RDF.type, OWL.FunctionalProperty))
            property_type_label = "Object Property (default)"
        
        # Añadir metadatos de la propiedad
        g.add((property_uri, RDFS.label, Literal(label_value, lang="en")))
        g.add((property_uri, RDFS.comment, Literal(f"{property_type_label} para el atributo {attribute}")))
        
        # Establecer dominio (la clase de la tabla correspondiente)
        if table in table_classes:
            g.add((property_uri, RDFS.domain, table_classes[table]))
        
        print(f"Propiedad creada: {property_name} ({property_type_label}) para tabla {table}")
    
    # Guardar la ontología principal
    try:
        # Guardar solo en formato OWL/XML
        g.serialize(destination=output_file_path, format='xml')
        print(f"Ontología principal guardada en formato OWL/XML: {output_file_path}")
        
    except Exception as e:
        print(f"Error al guardar la ontología principal: {e}")
        return
    
    # Guardar la ontología de diccionario
    try:
        # Generar nombre del archivo del diccionario
        dict_output_path = output_file_path.replace('.owl', '_dic.owl')
        dict_g.serialize(destination=dict_output_path, format='xml')
        print(f"Ontología de diccionario guardada en formato OWL/XML: {dict_output_path}")
        
    except Exception as e:
        print(f"Error al guardar la ontología de diccionario: {e}")
        return
    
    # Estadísticas
    print(f"\n--- ESTADÍSTICAS ---")
    print(f"Tablas procesadas: {len(unique_tables)}")
    print(f"Clases creadas en ontología principal: {len(unique_tables) + 1}")  # +1 por la clase Tables
    print(f"Clases de grupo creadas en diccionario: {len(dictionary_group_classes)}")
    print(f"Clases de rango creadas en diccionario: {len(range_classes_created)}")
    print(f"Total clases en diccionario: {len(dictionary_group_classes) + len(range_classes_created) + 2}")  # +2 por Dictionary e InstanceIdentifier
    print(f"Propiedades de atributos creadas: {len(properties_created)}")
    print(f"Total propiedades: {len(properties_created)}")
    print(f"Total de triples RDF en ontología principal: {len(g)}")
    print(f"Total de triples RDF en diccionario: {len(dict_g)}")
    
    return g, dict_g


def generate_mappings_ontology_from_csv(csv_file_path, output_file_path, ontology_name=None):
    """
    Genera una ontología OWL basada en mappings entre modelos
    
    Args:
        csv_file_path (str): Ruta al archivo CSV de mappings
        output_file_path (str): Ruta donde guardar la ontología
        ontology_name (str): Nombre personalizado para la ontología (opcional)
    """
    
    # Leer el CSV
    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV de mappings leído exitosamente. {len(df)} filas encontradas.")
    except Exception as e:
        print(f"Error al leer el CSV de mappings: {e}")
        return
    
    # Verificar que las columnas necesarias existan (ajustando para el formato actual)
    expected_columns = 4
    if len(df.columns) < expected_columns:
        print(f"Error: El archivo CSV debe tener al menos {expected_columns} columnas")
        return
    
    # Renombrar columnas para claridad
    df.columns = ['standard_attribute', 'standard_table', 'local_attribute', 'local_table']
    
    # Filtrar solo filas donde hay mappings (columnas 3 y 4 no están vacías)
    mappings_df = df.dropna(subset=['local_attribute', 'local_table'])
    
    if mappings_df.empty:
        print("No se encontraron mappings válidos en el archivo CSV")
        return
    
    print(f"Se encontraron {len(mappings_df)} mappings válidos")
    
    # Generar nombre de ontología basado en el archivo CSV si no se proporciona
    if ontology_name is None:
        csv_filename = os.path.basename(csv_file_path)
        ontology_name = os.path.splitext(csv_filename)[0]
    
    csv_filename = os.path.basename(csv_file_path)
    
    # Configurar namespaces
    base_uri = f"http://infmed.fcrb.es/ontologias/{ontology_name.lower()}"
    namespace_uri = f"{base_uri}#"
    
    # Namespace para el modelo local
    local_base_uri = f"http://infmed.fcrb.es/ontologias/localdb"
    local_namespace_uri = f"{local_base_uri}#"
    
    # Namespace para el modelo estándar
    standard_base_uri = f"http://infmed.fcrb.es/ontologias/standard_model"
    standard_namespace_uri = f"{standard_base_uri}#"
    
    # Definir namespaces
    ns = Namespace(namespace_uri)
    local_ns = Namespace(local_namespace_uri)
    standard_ns = Namespace(standard_namespace_uri)
    
    # Crear el grafo RDF
    g = Graph()
    
    # Configurar namespaces con bind
    g.bind(ontology_name.lower(), ns)
    g.bind("local", local_ns)
    g.bind("standard", standard_ns)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    
    # Crear la ontología
    ontology_uri = URIRef(base_uri)
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal(f"{ontology_name} Mappings Ontology")))
    g.add((ontology_uri, RDFS.comment, Literal(f"Ontología de mappings generada automáticamente basada en {csv_filename}")))
    
    # Importar las ontologías referenciadas
    g.add((ontology_uri, OWL.imports, URIRef(local_base_uri)))
    g.add((ontology_uri, OWL.imports, URIRef(standard_base_uri)))
    
    # Crear las propiedades equivalentes
    equivalent_properties_created = set()
    
    for _, row in mappings_df.iterrows():
        local_attribute = row['local_attribute']
        local_table = row['local_table']
        standard_attribute = row['standard_attribute']
        standard_table = row['standard_table']
        
        # Generar URIs para las propiedades usando la misma nomenclatura que la función original
        # Propiedad local: "has" + LocalTable + LocalAttribute
        local_table_class_name = to_pascal_case(local_table)
        local_attribute_name = to_pascal_case(local_attribute)
        local_property_name = "has" + local_table_class_name + local_attribute_name
        local_property_uri = local_ns[local_property_name]
        
        # Propiedad estándar: "has" + StandardTable + StandardAttribute
        standard_table_class_name = to_pascal_case(standard_table)
        standard_attribute_name = to_pascal_case(standard_attribute)
        standard_property_name = "has" + standard_table_class_name + standard_attribute_name
        standard_property_uri = standard_ns[standard_property_name]
        
        # Crear un identificador único para el par de propiedades
        mapping_key = f"{local_property_name}-{standard_property_name}"
        
        # Evitar duplicados
        if mapping_key in equivalent_properties_created:
            continue
        
        equivalent_properties_created.add(mapping_key)
        
        # Crear la relación de equivalencia bidireccional
        g.add((local_property_uri, OWL.equivalentProperty, standard_property_uri))
        g.add((standard_property_uri, OWL.equivalentProperty, local_property_uri))
        
        # Añadir metadatos para documentar el mapping
        mapping_comment = f"Equivalencia entre {local_attribute} de {local_table} y {standard_attribute} de {standard_table}"
        g.add((local_property_uri, RDFS.comment, Literal(mapping_comment)))
        g.add((standard_property_uri, RDFS.comment, Literal(mapping_comment)))
        
        print(f"Equivalencia creada: {local_property_name} == {standard_property_name}")
    
    # Guardar la ontología
    try:
        g.serialize(destination=output_file_path, format='xml')
        print(f"Ontología de mappings guardada en formato OWL/XML: {output_file_path}")
        
    except Exception as e:
        print(f"Error al guardar la ontología de mappings: {e}")
        return
    
    # Estadísticas
    print(f"\n--- ESTADÍSTICAS DE MAPPINGS ---")
    print(f"Mappings procesados: {len(mappings_df)}")
    print(f"Propiedades equivalentes creadas: {len(equivalent_properties_created)}")
    print(f"Total de triples RDF: {len(g)}")
    
    return g


def process_csv_files(input_folder="input", output_folder="output"):
    """
    Procesa todos los archivos CSV en la carpeta de entrada y genera ontologías
    
    Args:
        input_folder (str): Carpeta que contiene los archivos CSV
        output_folder (str): Carpeta donde guardar las ontologías generadas
    """
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Buscar archivos CSV en la carpeta de entrada
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"ERROR: No se encontraron archivos CSV en la carpeta '{input_folder}'")
        return
    
    print(f"Encontrados {len(csv_files)} archivo(s) CSV en '{input_folder}'")
    
    for csv_file in csv_files:
        print(f"\nProcesando: {os.path.basename(csv_file)}")
        
        # Generar nombre de archivo de salida
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = os.path.join(output_folder, f"{base_name}.owl")
        
        # Determinar si es un archivo de mappings
        if base_name.lower() == 'mappings':
            # Procesar archivo de mappings
            print("Detectado archivo de mappings - generando ontología de equivalencias")
            ontology_result = generate_mappings_ontology_from_csv(csv_file, output_file, base_name)
        else:
            # Procesar archivo normal
            print("Procesando archivo estándar - generando ontología normal")
            ontology_result = generate_ontology_from_csv(csv_file, output_file, base_name)
        
        if ontology_result:
            print(f"Ontología generada: {output_file}")
        else:
            print(f"ERROR: Error procesando: {csv_file}")


def main():
    """Función principal"""
    print("=== GENERADOR GENERAL DE ONTOLOGÍAS ===")
    print("Procesando archivos CSV de la carpeta 'input'...")
    print("-" * 50)
    
    # Procesar todos los archivos CSV
    process_csv_files()
    
    print("\nProcesamiento completado!")


if __name__ == "__main__":
    main()
