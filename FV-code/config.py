# config.py
"""
Parámetros globales de la infraestructura numérica
--------------------------------------------------
NGHOST  : número de celdas fantasma necesarias por lado.
          Elige el máximo que pueda requerir cualquiera de
          tus esquemas de reconstrucción (MP5 y WENO-5 necesitan 3).
"""

NGHOST = 3
