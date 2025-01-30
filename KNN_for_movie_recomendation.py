import math
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def user_based(file):
    # Fonction pour le filtrage collaboratif utilisateur
    l = open('ml-100k/u.user', 'r')  # Ouverture du fichier d'utilisateurs
    m = open(file, 'r')  # Ouverture du fichier de données d'entrainement
    o = open('ml-100k/u.occupation', 'r')  # Ouverture du fichier des occupations
    h = open('ml-100k/u.item', 'r')  # Ouverture du fichier des items

    s = h.readlines()  # Lecture des données des items
    x = l.readlines()  # Lecture des données des utilisateurs
    y = o.readlines()  # Lecture des occupations des utilisateurs
    z = m.readlines()  # Lecture des données de notation (ratings)
    X = []  # Liste des caractéristiques des utilisateurs
    rating = np.zeros((len(x), len(s)))  # Matrice de notation (ratings) de taille (utilisateur, item)

    # Remplissage de la matrice de notation (ratings)
    for i in range(len(z)):
        q = z[i].split("\t")
        rating[int(q[0]) - 1][int(q[1]) - 1] = int(q[2])

    # Traitement des occupations des utilisateurs
    for i in range(len(y)):
        p = ''
        for j in range(len(y[i]) - 1):
            p += y[i][j]
        y[i] = p

    # Extraction des caractéristiques des utilisateurs
    for i in range(len(x)):
        r = x[i].split("|")
        if r[2] == 'M':
            X.append([int(r[1]), 1, y.index(r[3])])
        else:
            X.append([int(r[1]), 0, y.index(r[3])])

    R = []

    # Pour chaque utilisateur
    for i in range(len(x)):
        test_user = [X[i]]  # Utilisateur de test (celui à qui on recommande)

        knn = NearestNeighbors(n_neighbors=1, metric='cosine')  # Initialisation du modèle de knn
        knn.fit(X)  # Entrainement du modèle knn sur les données X

        # Recherche des utilisateurs les plus similaires
        distances, indices = knn.kneighbors(test_user)

        # Calcul des notations prédites pour chaque item
        mat_ratine = np.zeros(len(s))
        T = [indices[0][i] for i in range(len(indices[0]))]

        for i in range(1682):
            u = 0
            t = 0
            for j in T:
                u += rating[j][i]
                if rating[j][i] != 0:
                    t += 1
            if t != 0:
                mat_ratine[i] = u / t
        R.append(mat_ratine)

    # Fermeture des fichiers
    h.close()
    l.close()
    m.close()
    o.close()
    return R

def item_based(file):
    # Fonction pour le filtrage collaboratif basé sur les items
    ratings_df = pd.read_csv(file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    items_df = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1')

    R = []
    for user_id in range(ratings_df['user_id'].nunique()):
        # Filtrage des notations pour l'utilisateur actuel
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        high_ratings = user_ratings[user_ratings['rating'] > 3]  # On ne considère que les notations élevées

        # Obtention de la liste des items regardés par l'utilisateur
        watched_items = high_ratings['item_id'].tolist()

        # Calcul de la notation moyenne pour chaque catégorie d'items
        category_ratings = items_df.iloc[:, 5:24].values.astype(int)
        avg_ratings = np.zeros(19)
        for category_index in range(19):
            total_rating = sum(category_ratings[item_id][category_index] for item_id in watched_items if item_id < len(category_ratings))
            count = sum(1 for item_id in watched_items if item_id < len(category_ratings) and category_ratings[item_id][category_index] > 0)
            avg_ratings[category_index] = total_rating / count if count > 0 else 0

        # Utilisation de NearestNeighbors pour trouver les utilisateurs similaires en fonction des notations de catégorie
        knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn.fit(category_ratings)

        # Recherche des utilisateurs similaires
        distances, indices = knn.kneighbors([avg_ratings])
        similar_users = indices[0]

        R.append(similar_users)

    return R
def fin_mat(a, b):
    # Fonction pour la combinaison des résultats des approches utilisateur et item-based
    P = []
    for i in range(len(a)):
        R = []
        for j in range(len(b[i])):
            R.append(a[i][b[i][j]])
        P.append(R)
    return P

def test_mat(file):
    # Fonction pour lire les données de test et les convertir en matrice
    l = open(file, 'r')
    h = l.readlines()
    p = []
    q = []
    j = []
    for i in range(len(h)):
        p.append(int(h[i].split("\t")[0]))
        q.append(int(h[i].split("\t")[1]))
        j.append(int(h[i].split("\t")[2]))
    x, y = max(p), max(q)
    R = np.zeros((x, y))
    for i in range(len(p)):
        R[p[i] - 1][q[i] - 1] = j[i]

    return R, x, y

# Boucle pour tester les différents fichiers de données
def testRMSE():
    S = 0
    for i in range(1, 6):

        file_base = f'ml-100k/u{i}.base'
        file_test = f'ml-100k/u{i}.test'

        items_df = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1')

        # Exécution des approches utilisateur et item-based
        mat_user_based = user_based(file_base)
        mat_item_based = item_based(file_base)

        # Combinaison des résultats des deux approches
        R = fin_mat(mat_user_based, mat_item_based)
        I, m1, m2 = test_mat(file_test)

        # Création de la matrice finale(on prend les donner dont on a besoin dans R)
        V = np.zeros((m1, m2))
        for j in range(len(I)):
            for k in range(len(mat_item_based[j])):
                if mat_item_based[j][k] < m2:
                    V[j][mat_item_based[j][k]] = R[j][k]


        # Calcul de l'erreur quadratique moyenne (RMSE)
        RMSE = mean_squared_error(I, V)
        S+=math.sqrt(RMSE)
        print(f"l'erreur du test {i} est de : {math.sqrt(RMSE)}")

    print(f"la moyenne est {S/5}")

def recommendation(user_id):
    file_base = 'ml-100k/u.data'
    items_df = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1')
    # Exécution des approches utilisateur et item-based
    mat_user_based = user_based(file_base)
    mat_item_based = item_based(file_base)
    R = fin_mat(mat_user_based, mat_item_based)

    B=[]#matrice des recomendation sera de la forme [[film_name,film_url,prediction_rating],...]
    for i in range(len(R[user_id])):
        if R[user_id][i] > 3 :
            B.append([items_df[1][i],items_df[4][i],R[user_id][i]])
            print(f"on vous recommende le film {items_df[1][i]} avec une estimation {R[user_id][i]} pour plus d'info voir : {items_df[4][i]}")
    return B


testRMSE()
recommendation(365)