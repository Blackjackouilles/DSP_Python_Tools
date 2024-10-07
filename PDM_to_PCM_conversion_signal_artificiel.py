#code to convert PDM to PCM
#avec ceci, PDM_Generation() génère une sinosoïde PDM
#PDM_To_PCM(bits): la converti en PCM et créé un fichier output_signal.wav que l'on peut lire avec Audacity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, decimate
from scipy.io.wavfile import write
from scipy.fft import fft

# Conversion PDM vers PCM
def PDM_To_PCM(bits):
    nyq_rate = 1e6 / 2.
 
    # The cutoff frequency of the filter: 100Hz
    cutoff_hz = 100.0
    # Length of the filter (number of coefficients, i.e. the filter order + 1)
    numtaps = 512
    # Use firwin to create a lowpass FIR filter
    fir_coeff = firwin(numtaps, cutoff_hz/nyq_rate)
    # Use lfilter to filter the signal with the FIR filter
    pdm_filtered = lfilter(fir_coeff, 1.0, bits)

    # Visualiser le signal filtré
    plt.plot(pdm_filtered[0:10000])  # Afficher les premiers échantillons filtrés
    plt.title('Signal PDM filtré')
    plt.show()

    # Décimation du signal filtré pour ramener à 44.1 kHz
    target_sampling_rate = 44100  # Fréquence d'échantillonnage cible
    # Calculer le facteur de décimation
    decimation_factor_final = int(1e6 / target_sampling_rate)

    # Appliquer la décimation
    pdm_decimated = decimate(pdm_filtered, decimation_factor_final)

    # Visualiser le signal filtré décimé
    plt.plot(pdm_decimated[0:1000])  # Afficher les premiers échantillons filtrés et décimés
    plt.title('Signal PDM filtré')
    plt.show()

    # Représentaion fréquentielle avant filtrage
    # Appliquer la FFT
    pdm_fft = fft(bits)
    # Calculer les fréquences correspondantes
    frequencies = np.fft.fftfreq(len(bits), 1/1e6)
    # Magnitude du spectre
    magnitude = np.abs(pdm_fft)
    # Visualisation du spectre de fréquences
    plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
    plt.title('Spectre fréquentiel du signal PDM')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    # Représentaion fréquentielle après filtrage et decimation
    pcm_fft = fft(pdm_decimated)
    # Calculer les fréquences correspondantes
    frequencies = np.fft.fftfreq(len(pdm_decimated), 1/target_sampling_rate)
    # Magnitude du spectre
    magnitude = np.abs(pcm_fft)
    # Visualisation du spectre de fréquences
    plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
    plt.title('Spectre fréquentiel du signal PCM filtré et décimé')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    # Écrire le fichier WAV
    output_file = 'output_signal.wav'
    write(output_file, target_sampling_rate, pdm_decimated)

    return

# Visualisation d'un signal
def Display_Bit_Stream(bits):

    # Création de l'axe des abscisses
    x = range(1, len(bits) + 1)  # Crée une liste de 1 à la longueur de bits

    # Création du graphique
    plt.figure(figsize=(10, 5))
    plt.step(x, bits, where='post', linewidth=2, color='blue', label='Signal')
    plt.fill_between(x, bits, step='post', alpha=0.4)

    # Étiquettes et titre
    plt.title('Représentation des Bits')
    plt.xlabel('Index des Bits')
    plt.ylabel('Valeur du Bit (0 ou 1)')
    plt.xticks(x)  # Afficher chaque valeur de x sur l'axe des abscisses
    plt.yticks([0, 1])  # Afficher 0 et 1 sur l'axe des ordonnées
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.1, 1.1)  # Pour mieux voir les valeurs 0 et 1

    # Affichage du graphique
    plt.show()

    return

# Génération signal PDM
def PDM_Generation():
    # Paramètres
    frequency = 440  # Fréquence du signal en Hz
    duration = 5     # Durée en secondes
    sampling_rate = 1e6  # Fréquence d'échantillonnage en Hz

    # Création de l'échantillon
    t = np.arange(0, duration, 1/sampling_rate)  # Vecteur temps

    # Génération du signal sinusoïdal
    # np.sin(2 * np.pi * frequency * t) : Cette expression génère une onde sinusoïdale de fréquence frequency et de période t.
    # C'est le signal analogique de base, qui varie entre -1 et 1.
    #
    # + 1 : Cette opération déplace la sinusoïde vers le haut, ce qui la fait varier entre 0 et 2 au lieu de -1 et 1.
    #
    # 0.5 * ( ... ) : Cela réduit l'amplitude du signal pour qu'il varie entre 0 et 1. La multiplication par 0.5 normalise donc
    # l'amplitude du signal entre ces deux valeurs, ce qui est souvent requis pour certaines opérations de traitement numérique (par exemple, pour PDM).
    #
    # analog_signal est une sinusoïde de fréquence donnée, oscillant entre 0 et 1. Cela le rend plus facile à convertir en un signal numérique ou à moduler.
    analog_signal = 0.5 * (np.sin(2 * np.pi * frequency * t) + 1)  # Normalisé entre 0 et 1

    # Convertir le signal analogique en PDM
    # np.random.rand(len(analog_signal)) : Cela génère un tableau aléatoire de la même longueur que analog_signal, où chaque élément est un nombre aléatoire
    # compris entre 0 et 1. Ce tableau représente du "bruit" aléatoire ou une comparaison de seuil.
    #
    # analog_signal > np.random.rand(len(analog_signal)) : Cette comparaison crée un signal binaire (valeurs True ou False) en comparant chaque échantillon
    # du signal analogique à une valeur aléatoire correspondante. Si un échantillon de analog_signal est supérieur au nombre aléatoire à la même position dans
    # np.random.rand, il est converti en True (1 en PDM), sinon en False (0 en PDM). Cette technique permet de moduler la densité des impulsions en fonction de
    # la valeur de l'amplitude du signal analogique.
    #
    # .astype(np.uint8) : Cette méthode convertit les valeurs booléennes (True ou False) en entiers non signés de 8 bits (1 ou 0), créant ainsi un flux de bits
    # PDM (pdm_signal).

    # 3. Comparaison avec un nombre aléatoire
    """
    La comparaison avec un seuil aléatoire est une manière simple et efficace d'approximer cette modulation. Voici pourquoi :

    Imagine que le signal analogique à un instant donné est de 0.75 (amplitude élevée).
        On génère un nombre aléatoire entre 0 et 1.
        Si ce nombre est inférieur à 0.75, on met un 1, sinon un 0.
        Comme la valeur 0.75 est relativement élevée, il y a une forte probabilité que le nombre aléatoire soit inférieur à 0.75, donc on obtiendra plus de 1.

    À l'inverse, si le signal analogique à un instant donné est de 0.25 (amplitude faible),
        On compare avec un autre nombre aléatoire.
        Si le nombre aléatoire est inférieur à 0.25, on met un 1, sinon un 0.
        Comme la valeur 0.25 est basse, la probabilité que le nombre aléatoire soit inférieur à 0.25 est faible, donc on aura moins de 1 et plus de 0.
    """
    pdm_signal = (analog_signal > np.random.rand(len(analog_signal))).astype(np.uint8)
    
    # Visualiser le signal PDM
    plt.figure(figsize=(12, 6))
    plt.plot(pdm_signal[:1000])  # Afficher les 1000 premiers bits
    plt.title('Signal PDM de 440 Hz')
    plt.xlabel('Échantillons')
    plt.ylabel('Bits PDM')
    plt.grid()
    plt.show()
    
    return pdm_signal

PDM_Bit_stream = PDM_Generation()

PDM_To_PCM(PDM_Bit_stream)

print("End")