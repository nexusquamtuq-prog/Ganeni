# Ganeni

Implementación de referencia del Modelo Ganeni  
Marco computacional para el análisis de campos configuracionales,
redundancia estructural y transiciones críticas.

## Qué es Ganeni
Ganeni es un modelo que describe la dinámica de configuraciones en espacios
de alta dimensión mediante un campo vectorial de repulsión configuracional.

## Qué hay en este repositorio
- Implementación del campo Ganeni (`GaneniField`)
- Métricas: R, IRC, ICA
- Simulación de trayectorias y colapso
- Visualización del espacio configuracional

## Uso rápido
```python
field = GaneniField(dimension=10)
field.add_configurations(configs)
R = field.compute_redundancy_index()

