import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, decimate
from numpy.fft import fft, fftfreq

# Paramètres
Fe_PDM = 1_000_000  # Fréquence d'échantillonnage PDM
f_sin = 440         # Fréquence de la sinusoïde
N = Fe_PDM // f_sin # Nombre de points pour une période
t = np.arange(0, N*10) / Fe_PDM

# Création d'un signal sinusoïdal
sin_wave = 0.5 * (1 + np.sin(2 * np.pi * f_sin * t))

# Conversion en PDM (méthode basique par modulation sigma-delta)
pdm_signal = (sin_wave > np.random.rand(len(sin_wave))).astype(np.uint8)

# Filtrage pour la conversion en PCM
decimation_factor = 64
cutoff = 0.00000001  # Fréquence de coupure normalisée
numtaps = 128  # Nombre de taps du filtre FIR

# Création du filtre FIR passe-bas
lowpass_filter = firwin(numtaps, cutoff=0.00000001)

# Appliquer le filtre au signal PDM
pdm_filtered = lfilter(lowpass_filter, 1.0, pdm_signal)

# Décimation pour obtenir un signal PCM à une fréquence plus basse
pcm_signal = decimate(pdm_filtered, decimation_factor)

# Représentation temporelle et fréquentielle
def plot_time_freq(signal, Fe, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(signal[:10000])  # Affichage des 1000 premiers échantillons
    ax1.set_title(f"{title} - Signal temporel")
    ax1.set_xlabel("Échantillons")
    ax1.set_ylabel("Amplitude")

    # Spectre fréquentiel
    N = len(signal)
    freq = fftfreq(N, 1/Fe)[:N//2]
    fft_signal = np.abs(fft(signal))[:N//2]
    
    ax2.plot(freq, fft_signal)
    ax2.set_title(f"{title} - Spectre fréquentiel")
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# Représentation du signal PDM
plot_time_freq(pdm_signal, Fe_PDM, "Signal PDM")

# Représentation du signal PCM
Fe_PCM = Fe_PDM // decimation_factor
plot_time_freq(pcm_signal, Fe_PCM, "Signal PCM")
