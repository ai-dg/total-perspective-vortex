
https://physionet.org/content/eegmmidb/1.0.0/

| Run     | Task                  | Réel / Imaginaire | mouvement                 | T1 = ?      | T2 = ?      |
| ------- | --------------------- | ----------------- | ------------------------- | ----------- | ----------- |
| **R01** | Baseline yeux ouverts | —                 | aucun                     | —           | —           |
| **R02** | Baseline yeux fermés  | —                 | aucun                     | —           | —           |
| **R03** | **Task 1**            | **réel**          | main gauche / main droite | main gauche | main droite |
| **R04** | **Task 2**            | ⭐ **imaginaire**  | main gauche / main droite | main gauche | main droite |
| **R05** | **Task 3**            | réel              | mains / pieds             | deux mains  | deux pieds  |
| **R06** | **Task 4**            | ⭐ **imaginaire**  | mains / pieds             | deux mains  | deux pieds  |
| **R07** | Task 1                | réel              | main gauche / main droite | main gauche | main droite |
| **R08** | Task 2                | ⭐ imaginaire      | main gauche / main droite | main gauche | main droite |
| **R09** | Task 3                | réel              | mains / pieds             | deux mains  | deux pieds  |
| **R10** | Task 4                | ⭐ imaginaire      | mains / pieds             | deux mains  | deux pieds  |
| **R11** | Task 1                | réel              | main gauche / main droite | main gauche | main droite |
| **R12** | Task 2                | ⭐ imaginaire      | main gauche / main droite | main gauche | main droite |
| **R13** | Task 3                | réel              | mains / pieds             | deux mains  | deux pieds  |
| **R14** | Task 4                | ⭐ imaginaire      | mains / pieds             | deux mains  | deux pieds  |


| Experiment ID | Run |
| ------------- | --- |
| 0             | R03 |
| 1             | R04 |
| 2             | R05 |
| 3             | R06 |
| 4             | R07 |
| 5             | R08 |


| Bande     | Fréquence  | Signification               |
| --------- | ---------- | --------------------------- |
| **Delta** | 0.5 – 4 Hz | sommeil profond             |
| **Theta** | 4 – 8 Hz   | somnolence, méditation      |
| **Alpha** | 8 – 12 Hz  | repos, yeux fermés          |
| **Beta**  | 12 – 30 Hz | activité mentale, mouvement |
| **Gamma** | 30–100 Hz  | cognition complexe          |


Avec :

64 = 64 channels EEG

N = nombre total d’échantillons dans le run

Exemple réel :

160 Hz = 160 échantillons par seconde

un run dure ~2 minutes → 120 s
→ donc N ≈ 160 × 120 = 19 200 colonnes

[ 64 channels ]  x  [ ~19000 échantillons (dans le temps) ]



1 / 160 = 0.00625 s par échantillon


High-pass → laisse passer les hautes fréquences

Low-pass → laisse passer les basses fréquences

✔ highpass: 0.0 Hz

Ça veut dire qu’ils n’ont pas appliqué de filtre coupe-bas matériel.
Le dataset garde même les très basses fréquences (0–1 Hz), qui contiennent :

mouvement des yeux

respiration

dérive du signal

fluctuations lentes

✔ lowpass: 80.0 Hz

Ça veut dire que le matériel a coupé toutes les fréquences au-dessus de 80 Hz.

Car au-dessus de 80–100 Hz, l’EEG scalp est presque uniquement du bruit musculaire, pas de l’activité neuronale.

Donc la bande 0–80 Hz est ce qui reste dans les fichiers .edf.

👉 C’est le filtrage matériel, pas ton filtrage logiciel.


✔ 0–4 Hz = delta (sommeil lent)
✔ 4–8 Hz = theta
✔ 8–12 Hz = mu / alpha (super utile pour motor imagery !)
✔ 12–30 Hz = beta (encore plus utile !)
✔ 30–80 Hz = gamma (souvent bruit musculaire)


Tes données actuelles (non filtrées) contiennent :

dérive lente (0–1 Hz)

yeux clignés (1–5 Hz)

mouvements de tête

ondes alpha (8–12 Hz)

ondes beta (12–30 Hz)

bruit musculaire (30–60 Hz)

un peu de gamma (jusqu’à 80 Hz)

Paramètre	Signification
highpass: 0 Hz	pas de filtre coupe-bas matériel (conserve les très basses fréquences)
lowpass: 80 Hz	le matériel EEG coupe tout au-dessus de 80 Hz
raw.filter(8,30)	ton filtre logiciel → garde les fréquences motrices
raw.notch_filter(50)	retire le bruit électrique (optionnel)

1 Hz = 1 oscillation par seconde
🔹 La hauteur de ses sauts = amplitude = µV
🔹 Le nombre de sauts par seconde = fréquence = Hz

| **Bande**          | **Fréquence (Hz)** | **Nom / Fonction**                               | **Lien avec le mouvement (réel ou imaginaire)**                                          |
| ------------------ | ------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Delta**          | 0.5 – 4 Hz         | sommeil profond                                  | pas utile (bruit, dérive lente)                                                          |
| **Theta**          | 4 – 8 Hz           | relaxation, navigation, mémoire                  | faible lien (un peu d’imagerie motrice)                                                  |
| **Alpha (µ / Mu)** | **8 – 12 Hz**      | **rythme sensorimoteur (SMR)**                   | ⭐ **diminue fortement (ERD) quand tu imagines ou fais un mouvement**, surtout dans C3/C4 |
| **Beta**           | **12 – 30 Hz**     | activité motrice, contrôle fin, retour sensoriel | ⭐ **augmente (ERS) après ou pendant l’imagination/mouvement**, très utile pour CSP       |
| **Gamma**          | 30 – 80 Hz         | cognition haute fréquence                        | peu utile en scalp EEG, contaminé par EMG (muscles)                                      |
| **Haut Gamma**     | >80 Hz             | potentiel local (LFP)                            | non significatif en EEG classique (trop bruité)                                          |


| Électrode | Zone                 | Fonction principale                            | Côté   |
| --------- | -------------------- | ---------------------------------------------- | ------ |
| **C3**    | Cortex moteur gauche | Imagination du **mouvement de la main droite** | Gauche |
| **C4**    | Cortex moteur droit  | Imagination du **mouvement de la main gauche** | Droit  |
| **Cz**    | Ligne médiane        | Point central, contrôle tronc/jambes           | Centre |


| Électrode | Zone               | Rôle                                   |
| --------- | ------------------ | -------------------------------------- |
| **FC3**   | Pré-moteur gauche  | Préparation du mouvement (main droite) |
| **FC4**   | Pré-moteur droit   | Préparation du mouvement (main gauche) |
| **CP3**   | Post-moteur gauche | Retour sensoriel (main droite)         |
| **CP4**   | Post-moteur droit  | Retour sensoriel (main gauche)         |


| Électrode     | Zone               | Rôle                            |
| ------------- | ------------------ | ------------------------------- |
| **C1 / C2**   | proches C3/C4      | Variation latérale              |
| **FC1 / FC2** | pré-moteur médial  | Confirme la préparation motrice |
| **CP1 / CP2** | post-moteur médial | Intégration sensorielle         |
| **C5 / C6**   | périphérique       | Mouvement bras / épaule         |


| Catégorie                | Électrodes         | Rôle                                      |
| ------------------------ | ------------------ | ----------------------------------------- |
| **Critiques**            | C3, C4, Cz         | C3=main droite, C4=main gauche, Cz=centre |
| **Autour (importants)**  | FC3, FC4, CP3, CP4 | Pré-moteur et sensorimoteur               |
| **Périphériques utiles** | C1, C2, C5, C6     | Contributions latérales                   |
| **Renfort**              | FC1, FC2, CP1, CP2 | Contribution médiane                      |

---
```bash
def manual_cov(X):
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (Xc.shape[0] - 1)



C1 = np.cov(X, rowvar=False)
C2 = manual_cov(X)
np.allclose(C1, C2, atol=1e-8)
```

```bash
vals_np, vecs_np = np.linalg.eigh(C)   # C symétrique
vals_my, vecs_my = manual_eigendecomposition(C)
# eigenvalues
np.allclose(np.sort(vals_np), np.sort(vals_my), atol=1e-6)
for i in range(k):  # k premières valeurs propres
    v1 = vecs_np[:, i]
    v2 = vecs_my[:, i]
    cos = abs(np.dot(v1, v2))  # ≈ 1 si mêmes directions
    assert cos > 0.99
---
U_np, S_np, Vt_np = np.linalg.svd(X, full_matrices=False)
U_my, S_my, Vt_my = manual_svd(X)
X_rec = U_my @ np.diag(S_my) @ Vt_my
np.allclose(X, X_rec, atol=1e-6)
np.allclose(np.sort(S_np), np.sort(S_my), atol=1e-6)

```
---
```bash
W_np = csp_numpy.filters_   # (n_filters, n_channels)
W_my = csp_manual.filters_

# Normaliser et comparer quelques colonnes via |cos(angle)|
for i in range(n_filters):
    v1 = W_np[i] / np.linalg.norm(W_np[i])
    v2 = W_my[i] / np.linalg.norm(W_my[i])
    cos = abs(np.dot(v1, v2))
    print(i, cos)  # ≈ 1 si équivalent
np.allclose(features_numpy, features_manual, atol=1e-4)
pipe_np = Pipeline([("csp", csp_numpy), ("clf", LogisticRegression())])
pipe_my = Pipeline([("csp", csp_manual), ("clf", LogisticRegression())])

acc_np = cross_val_score(pipe_np, X_flat, y, cv=5).mean()
acc_my = cross_val_score(pipe_my, X_flat, y, cv=5).mean()

print(acc_np, acc_my)
```


---
## Results

temps (s)= sample / 160

672 samples / 160 Hz = 4.2 secondes


[ 672     0     3 ]
   │      │      └── code événement (3 = T2)
   │      └──────── col. inutilisée
   └─────────────── sample index


[event_sample, previous_event_code, new_event_code]


| Code   | Signification | Détails                                                                                                                                                   | Utilisé pour la classification main gauche/droite ? |
| ------ | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **T0** | **Repos**     | Le sujet ne bouge pas, il ne fait pas d'imagerie motrice                                                                                                  | ❌ Non (à ignorer pour la classification MI)         |
| **T1** | **Classe 1**  | *Selon le type de run :* <br>• Runs 3, 7, 11 : **Mouvement/Imagination de la main gauche** <br>• Runs 5, 9, 13 : **Mouvement/Imagination des deux mains** | ✔️ Oui pour main gauche (runs 3,7,11)               |
| **T2** | **Classe 2**  | *Selon le type de run :* <br>• Runs 3, 7, 11 : **Mouvement/Imagination de la main droite** <br>• Runs 5, 9, 13 : **Mouvement/Imagination des deux pieds** | ✔️ Oui pour main droite (runs 3,7,11)               |

| Code   | Interprétation générale | Interprétation spécifique (runs 3,4,7,8,11,12) |
| ------ | ----------------------- | ---------------------------------------------- |
| **T0** | Repos (rien)            | Aucun mouvement → ignorer                      |
| **T1** | Activité classe 1       | **Main gauche (réelle ou imagée)**             |
| **T2** | Activité classe 2       | **Main droite (réelle ou imagée)**             |


| Run    | Type                                | T1     | T2     | ✔️ Utilisable ? |
| ------ | ----------------------------------- | ------ | ------ | --------------- |
| **3**  | Mouvement réel main gauche/droite   | Gauche | Droite | ✔️              |
| **4**  | Imagination main gauche/droite      | Gauche | Droite | ✔️              |
| **5**  | Mouv. réel deux mains / deux pieds  | Mains  | Pieds  | ❌               |
| **6**  | Imagination deux mains / deux pieds | Mains  | Pieds  | ❌               |
| **7**  | Mouvement réel main gauche/droite   | Gauche | Droite | ✔️              |
| **8**  | Imagination main gauche/droite      | Gauche | Droite | ✔️              |
| **9**  | Mouv. réel deux mains / deux pieds  | Mains  | Pieds  | ❌               |
| **10** | Imagination deux mains / deux pieds | Mains  | Pieds  | ❌               |
| **11** | Mouvement réel main gauche/droite   | Gauche | Droite | ✔️              |
| **12** | Imagination main gauche/droite      | Gauche | Droite | ✔️              |
| **13** | Mains / pieds                       | Mains  | Pieds  | ❌               |
| **14** | Mains / pieds                       | Mains  | Pieds  | ❌               |


|--- T0 ---|====T2====|---T0---|====T2====|---T0---|====T2====|
 samples
 0        672        1328      4656      5312      5984    etc.


tmin = -0.5, tmax = 4.0


[0.5 sec avant T2] → [4 sec après T2]


https://mindbigdata.com/opendb/index.html

https://www.physionet.org/content/sleep-edfx/1.0.0/

---
# CSP Eigenvalues sorting

“Note that λ(c)_j ≥ 0 is the variance in condition c in the corresponding surrogate channel and λ(+)_j + λ(-)_j = 1.”

“Hence a large value λ(+)_j close to one indicates that the corresponding spatial filter w_j yields high variance in the positive condition and low variance in the negative condition.”

(source: p.4)

λ(+)_j grand → variance élevée pour la classe + → variance faible pour la classe −

λ(+)_j petit → variance élevée pour la classe − → variance faible pour la classe +


# CSP W matrix
📌 1. Page 4 — Juste après l’équation (5)

C’est ici que CSP est défini mathématiquement.

Citation :

“Let W denote the matrix in which the rows give the filters.”


➡️ Cela signifie :
chaque ligne de W = un filtre CSP = un eigenvector transposé

📌 2. Page 4 — La diagonalisation conjointe

Le PDF rappelle que :

𝑊
𝑇
Σ
(
+
)
𝑊
=
Λ
(
+
)
W
T
Σ(+)W=Λ(+)
𝑊
𝑇
Σ
(
−
)
𝑊
=
Λ
(
−
)
W
T
Σ(−)W=Λ(−)

Cela implique :

👉 W est formée des eigenvectors du problème généralisé

Σ
+
𝑤
=
𝜆
Σ
−
𝑤
Σ
+
w=λΣ
−
w

C’est exactement ce que tu as déjà résolu.

📌 3. Page 5 — Sélection des filtres extrêmes

Citation :

“The first and last few eigenvectors yield filters with maximal discriminative power.”


Ceci est le point essentiel pour construire W :

Prendre k eigenvectors associés aux plus petites eigenvalues

Prendre k eigenvectors associés aux plus grandes eigenvalues

Puis les empiler dans W.


## Obtains features csp signals

📘 1. Page 3 — La transformation CSP & les features log-variance

👉 C’est ici que la partie “W @ X puis log-variance” est décrite.

Sur la page 3, section “A. Optimization Principles” et surtout Fig. 2, tu trouves :

“Variance of the projected sources is used as features.”
“Logarithmic power (log-variance) is commonly used as feature vector.”


📌 C’est exactement ton code :

z_i = W @ X_i
f_i = np.log(np.var(z_i, axis=1))

📘 2. Page 4 — Construction du filtre W et projection du signal

Dans la phrase juste après l’équation (5), le PDF dit :

“W denotes the matrix whose rows give the filters.”


Cela veut dire :

chaque ligne de W = un filtre CSP

ce filtre est appliqué directement au signal EEG :

𝑍
=
𝑊
𝑋
Z=WX

C’est mot pour mot la projection que tu fais :

z_i = W @ X_i

📘 3. Page 5 — Sélection des filtres extrêmes (donc composition de W)

“The first and last few eigenvectors yield filters with maximal discriminative power.”


Cela justifie pourquoi W contient :

les k eigenvectors avec petits eigenvalues

les k eigenvectors avec grands eigenvalues

Ce qui donne W dans ton code précédent.


1️⃣ D’où viennent alors les “6 experiments” ?

Dans le PDF 42, “experiment 0…5” ne veut pas dire “il existe 6 types de tâches dans le dataset”, mais :

6 configurations de classification différentes construites à partir des 4 tâches moteurs.

Une façon très naturelle (et cohérente avec le protocole) de définir ces 6 expériences est par ex. :

Exp 0 : Task 1 – réel main gauche vs main droite
→ runs 3, 7, 11 (T1=LG, T2=RD)

Exp 1 : Task 2 – imaginer main gauche vs main droite
→ runs 4, 8, 12

Exp 2 : Task 1+2 combinés (réel+imag) gauche vs droite
→ runs 3,4,7,8,11,12 (T1=LG, T2=RD)

Exp 3 : Task 3 – réel poings vs pieds
→ runs 5, 9, 13 (T1=poings, T2=pieds)

Exp 4 : Task 4 – imaginer poings vs pieds
→ runs 6, 10, 14

Exp 5 : Task 3+4 combinés (réel+imag) poings vs pieds
→ runs 5,6,9,10,13,14 (T1=poings, T2=pieds)

👉 Dans tous les cas, chaque experiment reste un problème binaire T1 vs T2, simplement avec un choix différent de runs / conditions (réel / imagé / mix).

Tu n’utilises jamais T0 dans ces expériences.

À partir de la description que tu as collée :

runs 3,4,7,8,11,12 :

T1 = left fist, T2 = right fist

runs 5,6,9,10,13,14 :

T1 = both fists, T2 = both feet

Je te proposais de construire les 6 experiments ainsi :

Exp 0 : Task 1 réel (main G/D) → runs 3,7,11

Exp 1 : Task 2 imagé (main G/D) → runs 4,8,12

Exp 2 : Task 1+2 mix (G/D réel+imagé) → runs 3,4,7,8,11,12

Exp 3 : Task 3 réel (poings/pieds) → runs 5,9,13

Exp 4 : Task 4 imagé (poings/pieds) → runs 6,10,14

Exp 5 : Task 3+4 mix (poings/pieds réel+imagé) → runs 5,6,9,10,13,14



Exp 0 → Train = 3 ; Test = 7,11
Exp 1 → Train = 4 ; Test = 8,12
Exp 2 → Train = 3 ; Test = 4,7,8,11,12
Exp 3 → Train = 5 ; Test = 9,13
Exp 4 → Train = 6 ; Test = 10,14
Exp 5 → Train = 5 ; Test = 6,9,10,13,14


Notes CSP — Soutenance MyCSP (42)

## 1. Prétraitement
- Band-pass (ex. 8–30 Hz) pour isoler les rythmes moteurs (µ / β) → ERD/ERS.
- Découpage en epochs autour des événements (ex. −0.5s → 4s).
→ **But : augmenter le rapport signal/bruit et capturer l’activité liée à T1/T2.**

## 2. Covariances
- Pour chaque classe : calcul d’une matrice de covariance normalisée.
- Normalisation par la trace :
Σ ← Σ / trace(Σ)
→ **Covariance = information spatiale du cerveau + patterns propres à chaque tâche.**

## 3. Problème aux valeurs propres (CSP)
- Résolution : Σ₊ w = λ Σ₋ w
- λ = ratio de variance entre classes après projection.
- Grand λ → variance forte pour classe +
- Petit λ → variance forte pour classe −
→ **Les eigenvectors = directions spatiales optimales.**

## 4. Sélection des filtres CSP
- On trie les eigenvalues.
- On prend k plus petits eigenvectors + k plus grands eigenvectors.
→ Les valeurs du milieu ne sont pas discriminantes.
→ Matrice : W ∈ ℝ^{2k × n_channels}

## 5. Projection CSP
- Pour chaque epoch : zᵢ = W Xᵢ
- Dimensions : (2k filtres × n_samples)
→ Chaque ligne = un filtre spatial qui maximise/minimise la variance selon la classe.

## 6. Extraction des features
- fᵢ[j] = log(Var(zᵢ[j]))
→ **Variance = information discriminante.**
→ **Log = stabilisation, compression, meilleure linéarité.**

## 7. Classification
- Features → LogisticRegression (ou LDA).
→ **CSP + log(var) = séparation linéaire entre classes.**

## Résumé court
Je filtre le signal, j’extrais les epochs, je calcule les covariances par classe,
puis je résous un problème généralisé d’eigenvalues pour obtenir les directions
où la variance est maximale pour une classe et minimale pour l’autre.
Je sélectionne les eigenvectors extrêmes, je projette les epochs, j’extrais le
log-variance et j’entraîne un classifieur dessus.




# Common Spatial Patterns (CSP) – Eigenvalues & Eigenvectors  
## Formules, explications et résolution mathématique

Ce document décrit clairement les formules utilisées dans CSP pour la
décomposition en valeurs propres (eigenvalues) et vecteurs propres (eigenvectors),
dans le cadre du problème généralisé entre les matrices de covariance Σ⁺ et Σ⁻.

---

# 1. Matrices Σ⁺ et Σ⁻ (Covariances moyennes)

À partir des epochs EEG filtrés :

- Xi : epoch i (shape : channels × time)
- Yi : label (classe + ou -)

Pour chaque epoch, on calcule sa covariance spatiale :

```
Ci = Xi * Xiᵀ
```

Puis on normalise par la trace :

```
Ci_norm = Ci / trace(Ci)
```

On sépare les epochs par classe :

```
Σ⁺ = moyenne des Ci_norm pour la classe +
Σ⁻ = moyenne des Ci_norm pour la classe -
```

Ces deux matrices résument la structure spatiale de chaque classe.

---

# 2. Problème généralisé d’autovaleurs (Generalized Eigenvalue Problem)

Dans CSP, on résout :

```
Σ⁺ w = λ Σ⁻ w
```

Ce problème trouve des vecteurs w qui :

- maximisent la variance pour Σ⁺ tout en la minimisant pour Σ⁻
- ou l’inverse (pour les petits λ)

Ce sont les directions spatiales discriminantes.

---

# 3. Passage à une forme standard

Le problème généralisé équivaut à :

```
Σ⁻⁻¹ Σ⁺ w = λ w
```

Ce qui est une décomposition propre classique.

---

# 4. Décomposition en valeurs propres

La décomposition d’une matrice A en valeurs propres consiste à trouver :

```
A v = λ v
```

Les vecteurs v sont invariants par la transformation A (changés seulement en échelle).

En regroupant tous les eigenvectors :

```
A V = V Λ
```

avec :

- V : matrice des eigenvectors (colonnes)
- Λ : matrice diagonale des eigenvalues

Si V est inversible :

```
A = V Λ V⁻¹
```

---

# 5. Application au CSP

On applique la décomposition généralisée :

```
eigvals, eigvecs = eig(Σ⁺, Σ⁻)
```

où :

- eigvals : λ₁ … λₙ (shape : n,)
- eigvecs : matrice V contenant les eigenvectors (shape : n × n)

---

# 6. Tri des eigenvectors

Les eigenvalues sont triés par ordre décroissant :

```
λ₁ ≥ λ₂ ≥ ... ≥ λₙ
```

et on réorganise les eigenvectors en conséquence.

Les eigenvectors associés aux :

- plus grands λ → maximisent la variance pour la classe +
- plus petits λ → maximisent la variance pour la classe –

---

# 7. Matrice de projection CSP (W)

En pratique, on sélectionne les k vecteurs propres aux extrémités :

```
W = [v₁ ... v_k, v_(n−k+1) ... v_n]ᵀ
```

W est la matrice finale CSP utilisée pour projeter les signaux :

```
Z = W X
```

Z contient les canaux virtuels discriminants.

---

# 8. Résumé rapide (pour soutenance)

- Σ⁺ et Σ⁻ = covariances moyennes par classe  
- CSP résout : Σ⁺ w = λ Σ⁻ w  
- λ = ratio de variance entre classes  
- w = direction spatiale maximisant ou minimisant la variance  
- On trie les eigenvalues et on garde les plus extrêmes  
- W = matrice CSP  
- Z = features CSP



# Common Spatial Patterns (CSP) – Problème généralisé d’autovaleurs  
## Formules avec notation $$ ... $$ (LaTeX compatible Markdown)

Ce document présente les formules utilisées dans CSP pour résoudre le problème généralisé d’autovaleurs entre les matrices de covariance de deux classes, en utilisant la notation LaTeX avec `$$`.

---

# 1. Covariance spatiale normalisée

Pour chaque epoch \(X_i \in \mathbb{R}^{C \times T}\) :

$$
C_i = X_i X_i^T
$$

Normalisation :

$$
\widetilde{C}_i = \frac{C_i}{\operatorname{trace}(C_i)}
$$

---

# 2. Covariances moyennes des deux classes

Soit :

- \( \mathcal{I}_+ \) = indices des epochs de la classe +
- \( \mathcal{I}_- \) = indices des epochs de la classe −

Alors :

$$
\Sigma^{(+)} = \frac{1}{|\mathcal{I}_+|} \sum_{i \in \mathcal{I}_+} \widetilde{C}_i
$$

$$
\Sigma^{(-)} = \frac{1}{|\mathcal{I}_-|} \sum_{i \in \mathcal{I}_-} \widetilde{C}_i
$$

---

# 3. Problème généralisé d’autovaleurs du CSP

Le CSP résout :

$$
\Sigma^{(+)} w = \lambda \, \Sigma^{(-)} w
$$

---

# 4. Transformation en problème classique

Si \( \Sigma^{(-)} \) est inversible :

$$
\Sigma^{(-)-1} \Sigma^{(+)} w = \lambda w
$$

On définit alors la matrice :

$$
M = \Sigma^{(-)-1} \Sigma^{(+)}
$$

et on résout :

$$
M w = \lambda w
$$

---

# 5. Décomposition propre (Eigen decomposition)

On peut écrire :

$$
M = V \Lambda V^{-1}
$$

où :

- \( V = [w_1 \ w_2 \ \cdots \ w_C] \) est la matrice des eigenvectors
- \( \Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_C) \) est la matrice diagonale des eigenvalues

Chaque colonne de \(V\) est un vecteur propre \(w_i\), associé à un eigenvalue \(\lambda_i\).

---

# 6. Calcul via SciPy

En pratique, on utilise :

```python
eigvals, eigvecs = scipy.linalg.eig(S_plus, S_minus)
```

Ce qui résout numériquement :

$$
\Sigma^{(+)} w = \lambda \, \Sigma^{(-)} w
$$

---

# 7. Sélection des filtres spatiaux CSP

Les eigenvalues sont triées par ordre décroissant :

$$
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_C
$$

- les plus grands \(\lambda\) correspondent à des filtres où la variance est maximale pour la classe +
- les plus petits \(\lambda\) correspondent à des filtres où la variance est maximale pour la classe −

On construit la matrice CSP en sélectionnant les eigenvectors extrêmes :

$$
W =
\begin{bmatrix}
w_1^T \\
w_2^T \\
\vdots \\
w_k^T \\
w_{C-k+1}^T \\
\vdots \\
w_C^T
\end{bmatrix}
$$

---

# 8. Projection finale des données

Pour un epoch \( X \in \mathbb{R}^{C \times T} \), la projection CSP est :

$$
Z = W X
$$

Les lignes de \(Z\) sont les composantes CSP (canaux virtuels discriminants) qui serviront de features pour la classification.

---

# 9. Résumé compact

$$
\begin{aligned}
&\widetilde{C}_i = \frac{X_i X_i^T}{\operatorname{trace}(X_i X_i^T)} \\\\
&\Sigma^{(+)} = \frac{1}{N_+} \sum \widetilde{C}_i,
\quad
\Sigma^{(-)} = \frac{1}{N_-} \sum \widetilde{C}_i \\\\
&\Sigma^{(+)} w = \lambda \Sigma^{(-)} w \\\\
&M = \Sigma^{(-)-1} \Sigma^{(+)} \\\\
&M = V \Lambda V^{-1} \\\\
&W = \text{matrice formée des eigenvectors extrêmes} \\\\
&Z = W X
\end{aligned}
$$



# CSP – Résolution mathématique de `scipy.linalg.eig(A, B)`
## Comment SciPy calcule les eigenvalues et eigenvectors du problème généralisé

Ce document décrit **exactement** comment SciPy résout le problème généralisé d’autovaleurs :

$$
A w = \lambda B w
$$

où \(A\) et \(B\) sont les matrices de covariance moyennes des deux classes (64×64 dans CSP).

---

# 1. Problème généralisé d’autovaleurs

Le CSP demande de résoudre :

$$
A w = \lambda B w
$$

où :

- \(A = \Sigma^{(+)}\) = covariance moyenne de la classe +  
- \(B = \Sigma^{(-)}\) = covariance moyenne de la classe −  
- \(w\) = vecteur propre (eigenvector)  
- \(\lambda\) = valeur propre (eigenvalue), ratio de variance entre classes

---

# 2. Décomposition de Cholesky de \(B\)

Comme \(B\) est symétrique définie positive, SciPy calcule :

$$
B = L L^T
$$

où \(L\) est triangulaire inférieure.

Ce facteur servira à transformer le problème généralisé en un problème standard.

---

# 3. Transformation (« whitening ») pour éliminer \(B\)

On pose :

$$
M = L^{-1} A L^{-T}
$$

et le changement de variable :

$$
w = L^{-T} u
$$

Substitution dans l’équation générale :

\[
A w = \lambda B w
\]

donne après simplification :

$$
M u = \lambda u
$$

On a converti le problème généralisé en un **problème classique d’autovaleurs**.

---

# 4. Décomposition propre standard

SciPy diagonalise ensuite la matrice :

$$
M = V \Lambda V^{-1}
$$

Ce qui signifie que chaque colonne \(u_i\) de \(V\) vérifie :

$$
M u_i = \lambda_i u_i
$$

Les \(\lambda_i\) sont les eigenvalues du CSP.

---

# 5. Retour aux eigenvectors du problème généralisé

SciPy reconstruit ensuite les eigenvectors originaux via :

$$
w_i = L^{-T} u_i
$$

Ainsi :

$$
A w_i = \lambda_i B w_i
$$

La matrice finale des eigenvectors fournie par SciPy est :

$$
W = L^{-T} V
$$

Chaque colonne de \(W\) est un filtre spatial CSP.

---

# 6. Résultat de `scipy.linalg.eig(A, B)`

L'appel :

```python
eigvals, eigvecs = scipy.linalg.eig(A, B)
```

retourne directement :

- `eigvals` = \( (\lambda_1, \ldots, \lambda_C) \)
- `eigvecs` =  
  $$ W = [w_1 \ w_2 \ \cdots \ w_C] $$

tels que :

$$
A w_i = \lambda_i B w_i
$$

SciPy effectue **automatiquement** toutes les étapes :

1. Cholesky de \(B\)  
2. Construction de \(M = L^{-1} A L^{-T}\)  
3. Résolution du problème standard \(Mu = \lambda u\)  
4. Reconstruction \(w = L^{-T} u\)  

---

# 7. Résumé compact (pour soutenance)

$$
\begin{aligned}
& A w = \lambda B w \\
& B = L L^T \\
& M = L^{-1} A L^{-T} \\
& M u = \lambda u \\
& w = L^{-T} u \\
& W = [w_1 \ w_2 \ \cdots \ w_C]
\end{aligned}
$$

---

# Fichier prêt pour utilisation dans votre projet CSP.