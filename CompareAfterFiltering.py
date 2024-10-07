from numpy import sin, arange, pi
import numpy as np
from scipy.signal import lfilter, firwin
from pylab import figure, plt, grid, show
from numpy.fft import fft, fftfreq
 

sample_rate = 48000.

def PDM_Generation():
    # Paramètres
    frequency = 440  # Fréquence du signal en Hz
    duration = 5     # Durée en secondes
    sampling_rate = sample_rate  # Fréquence d'échantillonnage en Hz

    # Création de l'échantillon
    t = np.arange(0, duration, 1/sampling_rate)  # Vecteur temps

    analog_signal = 0.5 * (np.sin(2 * np.pi * frequency * t) + 1)  # Normalisé entre 0 et 1

    pdm_signal = (analog_signal > np.random.rand(len(analog_signal))).astype(np.uint8)
    
    # Visualiser le signal PDM
    plt.figure(figsize=(12, 6))
    plt.plot(pdm_signal)  # Afficher les 1000 premiers bits
    plt.title('Signal PDM de 440 Hz')
    plt.xlabel('Échantillons')
    plt.ylabel('Bits PDM')
    plt.grid()
    plt.show()
    
    return pdm_signal

#------------------------------------------------
# Create a signal for demonstration.
#------------------------------------------------
# 320 samples of (1000Hz + 15000 Hz) at 48 kHz
nsamples = sample_rate * 5
 
t = arange(nsamples) / sample_rate
signal = PDM_Generation()
 
#------------------------------------------------
# Create a FIR filter and apply it to signal.
#------------------------------------------------
# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.
 
# The cutoff frequency of the filter: 6KHz
cutoff_hz = 100.0
 
# Length of the filter (number of coefficients, i.e. the filter order + 1)
numtaps = 128
 
# Use firwin to create a lowpass FIR filter
fir_coeff = firwin(numtaps, cutoff_hz/nyq_rate)
 
# Use lfilter to filter the signal with the FIR filter
filtered_signal = lfilter(fir_coeff, 1.0, signal)
 
#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
 
# The first N-1 samples are "corrupted" by the initial conditions
warmup = numtaps - 1
 
# The phase delay of the filtered signal
delay = (warmup / 2) / sample_rate
 
figure(1)
# Plot the original signal
plt.plot(t, signal)
 
# Plot the filtered signal, shifted to compensate for the phase delay
plt.plot(t-delay, filtered_signal, 'r-')
 
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plt.plot(t[warmup:]-delay, filtered_signal[warmup:], 'g', linewidth=4)
 
grid(True)
 
show()

# Représentaion fréquentielle avant filtrage
pcm_fft = fft(signal)
# Calculer les fréquences correspondantes
frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
# Magnitude du spectre
magnitude = np.abs(pcm_fft)
# Visualisation du spectre de fréquences
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.title('Spectre fréquentiel du signal PCM')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Magnitude')
plt.show()

# Représentaion fréquentielle avant filtrage
pcm_fft = fft(filtered_signal)
# Calculer les fréquences correspondantes
frequencies = np.fft.fftfreq(len(filtered_signal), 1/sample_rate)
# Magnitude du spectre
magnitude = np.abs(pcm_fft)
# Visualisation du spectre de fréquences
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.title('Spectre fréquentiel du signal PCM')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Magnitude')
plt.show()

 
#------------------------------------------------
# Print values
#------------------------------------------------
def print_values(label, values):
    var = "float32_t %s[%d]" % (label, len(values))
    #print "%-30s = {%s}" % (var, ', '.join(["%+.10f" % x for x in values]))
 
print_values('signal', signal)
print_values('fir_coeff', fir_coeff)
print_values('filtered_signal', filtered_signal)