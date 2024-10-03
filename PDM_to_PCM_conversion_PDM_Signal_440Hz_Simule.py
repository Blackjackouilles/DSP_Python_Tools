#code to convert PDM to PCM
#avec ceci, PDM_Generation() génère une sinosoïde PDM
#PDM_To_PCM(bits): la converti en PCM et créé un fichier output_signal.wav que l'on peut lire avec Audacity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, decimate
from scipy.io.wavfile import write
from scipy.fft import fft

def PDM_Format_Extraction(file_path):
    # Lire le fichier CSV
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Le fichier spécifié n'a pas été trouvé.")
        return

    # Vérifier si le DataFrame a au moins deux colonnes
    if data.shape[1] < 2:
        print("Le fichier CSV doit contenir au moins deux colonnes.")
        return

    return data['Channel 1'].tolist()

def PDM_To_PCM(bits):
    # Paramètres du filtre
    decimation_factor = 64  # Exemple : facteur de décimation
    cutoff = 0.45 / decimation_factor  # Fréquence de coupure normalisée
    numtaps = 64  # Nombre de taps du filtre FIR

    # Créer un filtre FIR passe-bas
    lowpass_filter = firwin(numtaps, cutoff)

    # Appliquer le filtre au bit-stream PDM (lissage des bits)
    pdm_filtered = lfilter(lowpass_filter, 1.0, bits)

    # Visualiser le signal filtré
    plt.plot(pdm_filtered)  # Afficher les 500 premiers échantillons filtrés
    plt.title('Signal PDM filtré')
    plt.show()

    # Décimation du signal filtré pour ramener à 44.1 kHz
    target_sampling_rate = 44100  # Fréquence d'échantillonnage cible
    # Calculer le facteur de décimation
    decimation_factor_final = int(1e6 / target_sampling_rate)

    # Appliquer la décimation
    pdm_decimated = decimate(pdm_filtered, decimation_factor_final)

    # Visualiser le signal filtré décimé
    # Visualiser le signal filtré
    plt.plot(pdm_decimated)  # Afficher les 500 premiers échantillons filtrés
    plt.title('Signal PDM filtré')
    plt.show()

    # Représentaion fréquentielle avant filtrage
    # Appliquer la FFT
    pcm_fft = fft(bits)
    # Calculer les fréquences correspondantes
    frequencies = np.fft.fftfreq(len(bits), 1/1e6)
    # Magnitude du spectre
    magnitude = np.abs(pcm_fft)
    # Visualisation du spectre de fréquences
    plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
    plt.title('Spectre fréquentiel du signal PCM')
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
    plt.title('Spectre fréquentiel du signal PCM')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    # Écrire le fichier WAV
    output_file = 'output_signal.wav'
    write(output_file, target_sampling_rate, pdm_decimated)

    return

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

def PDM_Generation():
    # Paramètres
    frequency = 440  # Fréquence du signal en Hz
    duration = 5     # Durée en secondes
    sampling_rate = 1e6  # Fréquence d'échantillonnage en Hz

    # Création de l'échantillon
    t = np.arange(0, duration, 1/sampling_rate)  # Vecteur temps
    # Génération du signal sinusoïdal
    analog_signal = 0.5 * (np.sin(2 * np.pi * frequency * t) + 1)  # Normalisé entre 0 et 1

    # Convertir le signal analogique en PDM
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

PDM_Bit_stream = [0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                  0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,
                  0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                  0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1]

Display_Bit_Stream(PDM_Bit_stream) # Limit to [0:500]

PDM_To_PCM(PDM_Bit_stream)

PDM_Bit_stream = PDM_Generation()

PDM_To_PCM(PDM_Bit_stream)

print("End")