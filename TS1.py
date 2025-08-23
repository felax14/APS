# -*- coding: utf-8 -*-
"""

TS1: Síntesis de señales

Author: FELIPE BERGANZA 

1) Sintetizar y graficar:

Una señal sinusoidal de 2KHz.
Misma señal amplificada y desfazada en π/2.
Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
Señal anterior recortada al 75% de su potencia.
Una señal cuadrada de 4KHz.
Un pulso rectangular de 10ms.
En cada caso indique tiempo entre muestras, número de muestras y potencia.

2) Verificar ortogonalidad entre la primera señal y las demás.

3) Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás.

3) Dada la siguiente propiedad trigonométrica:

2⋅ sin(α)⋅ sin(β) = cos(α-β)-cos(α+β)

Demostrar la igualdad
Mostrar que la igualdad se cumple con señales sinosoidales, considerando α=ω⋅t, el doble de β (Use la frecuencia que desee).
Bonus
4) Graficar la temperatura del procesador de tu computadora en tiempo real.

Suponiendo distancia entre muestras constante
Considerando el tiempo de la muestra tomada
5) Bajar un wav de freesoung.org, graficarlo y calcular la energía


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import psutil
import time

# =============================================================================
# 1) SÍNTESIS DE SEÑALES
# =============================================================================

# Parámetros generales
fs = 44100  # Frecuencia de muestreo [Hz]
t_total = 0.01  # Tiempo total de simulación [s] (10 ms)
nn = int(fs * t_total)  # Número de muestras
Ts = 1/fs  # Tiempo entre muestras [s]

print(f"Tiempo entre muestras: {Ts*1000:.4f} ms")
print(f"Número de muestras: {nn}")

# Frecuencia fundamental
f1 = 2000  # 2 kHz

# 1.1 Defino la funcion seno
def funcion_sen(vmax, dc, ff, ph, nn, fs):
    t_total = nn / fs  # Tiempo total de muestreo
    tt = np.linspace(0, t_total, nn, endpoint=False)  # Vector de tiempo
    
    # Creo la señal senoidal
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    
    return tt, xx

# 1.1 Señal sinusoidal de 2kHz
tt, xx1 = funcion_sen(vmax=1, dc=0, ff=f1, ph=0, nn=nn, fs=fs)
potencia_xx1 = np.mean(xx1**2)
print(f"\n1. Señal sinusoidal de 2kHz - Potencia: {potencia_xx1:.4f}")

# 1.2 Señal amplificada con un factor de 2 y desfasada en π/2
tt, xx2 = funcion_sen(vmax=2, dc=0, ff=f1, ph=np.pi/2, nn=nn, fs=fs)
potencia_xx2 = np.mean(xx2**2)
print(f"2. Señal amplificada y desfasada - Potencia: {potencia_xx2:.4f}")

# 1.3 Señal modulada en amplitud
# Creo la señal moduladora (mitad de frecuencia)
tt, portadora = funcion_sen(vmax=1, dc=0, ff=f1/2, ph=0, nn=nn, fs=fs)
modulacion = xx1 * portadora
potencia_modulacion = np.mean(modulacion**2)
print(f"3. Señal modulada AM - Potencia: {potencia_modulacion:.4f}")

# 1.4 Señal recortada al 75% de su potencia
# Calculo amplitud para 75% de potencia
amplitud_pico_original = np.max(np.abs(xx1))
potencia_original = (amplitud_pico_original**2) / 2
amplitud_pico_deseada = np.sqrt(2 * 0.75 * potencia_original)

# Recorto la señal
senal_recortada = np.clip(xx1, -amplitud_pico_deseada, amplitud_pico_deseada)
potencia_recortada = np.mean(senal_recortada**2)
print(f"4. Señal recortada al 75% - Potencia: {potencia_recortada:.4f}")

# 1.5 Función para crear una onda cuadrada de 4kHz
def funcion_cuadrada(vmax, dc, ff, nn, fs):
    t_total = nn / fs
    tt = np.linspace(0, t_total, nn, endpoint=False)
    
    periodo = 1 / ff
    estado = (tt % periodo) < (0.5 * periodo)  # ciclo de trabajo de 50%
    xx = np.where(estado, vmax, -vmax) + dc
    
    return tt, xx

# Creo la onda cuadrada de 4kHz
tt, xx4 = funcion_cuadrada(vmax=1, dc=0, ff=4000, nn=nn, fs=fs)
potencia_xx4 = np.mean(xx4**2)
print(f"5. Onda cuadrada 4kHz - Potencia: {potencia_xx4:.4f}")

# 1.6 Función para crear un pulso rectangular de 10ms
def pulso_rectangular(amplitud, duracion_total, fs, duracion_pulso, inicio=0):
    nn = int(fs * duracion_total)
    tt = np.linspace(0, duracion_total, nn, endpoint=False)
    
    pulso = np.zeros_like(tt)
    mask_pulso = (tt >= inicio) & (tt <= inicio + duracion_pulso)
    pulso[mask_pulso] = amplitud
    
    return tt, pulso

# Genero el pulso de 10 ms con la misma frecuencia de muestreo
tt_pulso, pulso_10ms = pulso_rectangular(amplitud=1, duracion_total=0.03, fs=fs, 
                                        duracion_pulso=0.01, inicio=0.005)
potencia_pulso = np.mean(pulso_10ms**2)
print(f"6. Pulso rectangular 10ms - Potencia: {potencia_pulso:.4f}")

# =============================================================================
# GRÁFICOS INDIVIDUALES PARA CADA SEÑAL
# =============================================================================

# 1.1 Señal sinusoidal de 2kHz
plt.figure(figsize=(10, 6))
plt.plot(tt[:1000], xx1[:1000], 'b-', linewidth=2)
plt.title('1.1 Señal sinusoidal de 2kHz', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.2 Señal amplificada y desfasada
plt.figure(figsize=(10, 6))
plt.plot(tt[:1000], xx2[:1000], 'r-', linewidth=2)
plt.title('1.2 Señal amplificada y desfasada π/2', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.3 Señal modulada AM
plt.figure(figsize=(10, 6))
plt.plot(tt[:2000], modulacion[:2000], 'g-', linewidth=2)
plt.title('1.3 Señal modulada AM', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.4 Señal recortada al 75%
plt.figure(figsize=(10, 6))
plt.plot(tt[:1000], senal_recortada[:1000], 'm-', linewidth=2)
plt.title('1.4 Señal recortada al 75% de potencia', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.5 Onda cuadrada de 4kHz
plt.figure(figsize=(10, 6))
plt.plot(tt[:2000], xx4[:2000], 'c-', linewidth=2)
plt.title('1.5 Onda cuadrada de 4kHz', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 1.6 Pulso rectangular de 10ms
plt.figure(figsize=(10, 6))
plt.plot(tt_pulso, pulso_10ms, 'orange', linewidth=2)
plt.title('1.6 Pulso rectangular de 10ms', fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# GRÁFICOS COMBINADOS SOLICITADOS
# =============================================================================

# Gráfico combinado: 1.1 y 1.3 en el mismo gráfico
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(tt[:1000], xx1[:1000], 'b-', linewidth=2, label='Señal original (2kHz)')
plt.plot(tt[:1000], modulacion[:1000], 'g-', linewidth=2, label='Señal modulada AM')
plt.title('Comparación: Señal original vs Modulada AM', fontsize=12)
plt.xlabel('Tiempo [s]', fontsize=10)
plt.ylabel('Amplitud', fontsize=10)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tt[:1000], xx1[:1000], 'b-', linewidth=2, label='Señal original')
plt.plot(tt[:1000], senal_recortada[:1000], 'm-', linewidth=2, label='Señal recortada 75%')
plt.title('Comparación: Señal original vs Recortada', fontsize=12)
plt.xlabel('Tiempo [s]', fontsize=10)
plt.ylabel('Amplitud', fontsize=10)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('Comparación de Señales: 1.1 con 1.3 y 1.1 con 1.4', fontsize=14, y=1.02)
plt.show()

# =============================================================================
# 2) VERIFICAR ORTOGONALIDAD
# =============================================================================

def verificar_ortogonales(senal1, senal2, umbral=1e-10, verbose=True):
    if len(senal1) != len(senal2):
        if verbose:
            print("❌ ERROR: Tamaños diferentes")
        return False
    
    producto = np.dot(senal1.flatten(), senal2.flatten())
    ortogonal = abs(producto) < umbral
    
    if verbose:
        if ortogonal:
            print(f"✅ SÍ son ortogonales (producto = {producto:.2e})")
        else:
            print(f"❌ NO son ortogonales (producto = {producto:.2e})")
    
    return ortogonal

print("\n=== VERIFICACIÓN DE ORTOGONALIDAD ===")
print("1. xx1 vs xx1:")
verificar_ortogonales(xx1, xx1)

print("\n2. xx1 vs xx2 (seno y coseno):")
verificar_ortogonales(xx1, xx2)

print("\n3. xx1 vs modulación:")
min_len = min(len(xx1), len(modulacion))
verificar_ortogonales(xx1[:min_len], modulacion[:min_len])

print("\n4. xx1 vs señal recortada:")
verificar_ortogonales(xx1, senal_recortada)

print("\n5. xx1 vs xx4 (onda cuadrada 4kHz):")
min_len = min(len(xx1), len(xx4))
verificar_ortogonales(xx1[:min_len], xx4[:min_len])

print("\n6. xx1 vs pulso 10ms:")
pulso_resized = np.interp(np.linspace(0, 1, len(xx1)), 
                         np.linspace(0, 1, len(pulso_10ms)), 
                         pulso_10ms.flatten())
verificar_ortogonales(xx1, pulso_resized)

# =============================================================================
# 3) CORRELACIONES
# =============================================================================

def calcular_correlacion(senal1, senal2=None):
    if senal2 is None:
        # Autocorrelación
        correlacion = np.correlate(senal1.flatten(), senal1.flatten(), mode='full')
    else:
        # Correlación cruzada
        min_len = min(len(senal1), len(senal2))
        correlacion = np.correlate(senal1[:min_len].flatten(), 
                                  senal2[:min_len].flatten(), mode='full')
    
    # Normalizar
    correlacion = correlacion / np.max(np.abs(correlacion))
    return correlacion

# Calcular correlaciones
autocorr_xx1 = calcular_correlacion(xx1)
corr_xx1_xx2 = calcular_correlacion(xx1, xx2)
corr_xx1_mod = calcular_correlacion(xx1, modulacion)
corr_xx1_xx4 = calcular_correlacion(xx1, xx4)

# Graficar correlaciones
lags = np.arange(-len(autocorr_xx1)//2 + 1, len(autocorr_xx1)//2 + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(lags, autocorr_xx1)
plt.title('Autocorrelación de la señal sinusoidal de 2kHz')
plt.xlabel('Desplazamiento')
plt.ylabel('Correlación')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(lags, corr_xx1_xx2)
plt.title('Correlación entre señal 1 y señal 2')
plt.xlabel('Desplazamiento')
plt.ylabel('Correlación')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(lags, corr_xx1_mod)
plt.title('Correlación entre señal 1 y modulación AM')
plt.xlabel('Desplazamiento')
plt.ylabel('Correlación')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(lags, corr_xx1_xx4)
plt.title('Correlación entre señal 1 y onda cuadrada')
plt.xlabel('Desplazamiento')
plt.ylabel('Correlación')
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# 3) DEMOSTRACIÓN PROPIEDAD TRIGONOMÉTRICA
# =============================================================================

print("\n=== DEMOSTRACIÓN PROPIEDAD TRIGONOMÉTRICA ===")
print("2⋅sin(α)⋅sin(β) = cos(α-β) - cos(α+β)")

# Definir parámetros
frecuencia = 1000  # 1 kHz
t = np.linspace(0, 0.01, 1000)  # 10 ms
alpha = 2 * np.pi * frecuencia * t
beta = 2 * np.pi * frecuencia/2 * t  # β es la mitad de α

# Calcular ambos lados de la ecuación
lado_izq = 2 * np.sin(alpha) * np.sin(beta)
lado_der = np.cos(alpha - beta) - np.cos(alpha + beta)

# Verificar igualdad
diferencia = np.max(np.abs(lado_izq - lado_der))
print(f"Diferencia máxima entre ambos lados: {diferencia:.2e}")

if diferencia < 1e-10:
    print("✅ La igualdad se cumple")
else:
    print("❌ La igualdad NO se cumple")

# Graficar
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, lado_izq, label='2⋅sin(α)⋅sin(β)')
plt.plot(t, lado_der, '--', label='cos(α-β)-cos(α+β)')
plt.title('Comparación de ambos lados de la ecuación')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, lado_izq - lado_der)
plt.title('Diferencia entre ambos lados')
plt.xlabel('Tiempo [s]')
plt.ylabel('Diferencia')
plt.grid(True)

plt.tight_layout()
plt.show()



