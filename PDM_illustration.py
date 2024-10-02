#code to visualise PDM bit-stream
import pandas as pd
import matplotlib.pyplot as plt

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

# Exemple d'utilisation
file_path = 'PDM_signal_5_ms.csv'  # Remplace par le chemin de ton fichier CSV

bits = PDM_Format_Extraction(file_path)

#test de visualisaiton d'un bit-stream
#bits = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]

Display_Bit_Stream(bits[0:500])

print("End")