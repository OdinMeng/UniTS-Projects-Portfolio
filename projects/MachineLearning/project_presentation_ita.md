---
marp: true
---

<style>
img[alt~='center'] {
display: block;
margin-left: auto;
margin-right: auto;
}
</style>

<style>
    .container{
        display: flex;
    }
    .col{
        flex: 1;
    }
</style>


<div>
<center> 

# PRESENTAZIONE PROGETTO MACHINE LEARNING
**DINO MENG [SM3201466]**

</center>
</div>


---

# Argomenti:

1. Idea del clustering spettrale, richiami su SVD
2. Algoritmo di clustering spettrale e variante basata su SVD
3. Esperimenti

---

# SVD (Richiami)

<div class="container">

<div class="col">

La *SVD* (Singular-Value Decomposition) è una tecnica nota di decomposizione di matrici basata sulla decomposizione spettrale

$$
A = U\Sigma V^T
$$

Applicazioni:

* *"Low-Rank Matrix Approximation"* (Teorema di Eckart–Young)
* Compressione delle immagini
* Riduzione della dimensionalità
* **Vedremo**: Clustering

</div>

<div class="col">

![center](./presentation_images/Singular_value_decomposition_visualisation.svg.png)

</div>
</div>

---
# Clustering spettrale (Formulazione del Problema)

**Problema**: Determinare un $k$-partizionamento del grafo $G=(V,E)$, minimizzando i tagli da effettuare

![width:900px center](./presentation_images/spectral_idea.png)

---

# Clustering spettrale

![center](./presentation_images/image.png)

> *Tratto da: Ulrike von Luxburg, A Tutorial on Spectral Clustering*

---

# Clustering spettrale (Osservazione)

> **Osservazione**: si dimostra che l'algoritmo descritto in precedenza risolve un opportuno rilassamento del problema dei minimi $k$-cut del grafo $G$

![width:450px center](./presentation_images/Minimum_k-cut.svg.png)

---

# Clustering spettrale basato su SVD (Idea)

**IDEA.** Invece di trovare gli autovalori del Laplaciano $L = D-A$, diagonalizzare le matrici $A^T A$ e $AA^T$ (determinando così una SVD di $A$)

Variante proposta da Zhixian Jia, nell'articolo "The reaserch on parameters of spectral clustering based on SVD" (2013)

---

# Clustering spettrale basato su SVD

Parametri:

* Numero di cluster $k$
* Numero di vettori singolari sinistri da considerare $l$ (di solito $l=k$)

Algoritmo:

1. Creare la matrice di somiglianza $A$ con un metodo a piacere
2. Determinare una SVD di $A$, cioè $A = V \Sigma U^T$
3. Prendere le prime $l$ colonne di $V$, e denotarle con $V'$
4. Eseguire *k-means* su $V'$, i cluster risultanti costituiscono l'output

---

# Esperimenti: Introduzione, obbiettivi

Tre esperimenti principali, basati su dataset generati sinteticamente:

1. Confrontare il clustering spettrale e la sua variante basata sulla SVD. Due metodi di costruzione del grafo $(V,E) \sim A$: kernel gaussiano (grafo pesato e completamente connesso) e KNN-graph.
2. Eseguire la *SVD-based Spectral Clustering* su dataset non lineari (variando i parametri)
3. Dedurre il parametro $k$ dallo studio della matrice dei valori singolari $\Sigma$

---

# Dataset sintetici

![center](./presentation_images/example_1.png)

---

# Dataset sintetici

![center](./presentation_images/non_linearly_sep.png)

---

# Dataset sintetici

![center](./presentation_images/example_3.png)

---

# Esperimento 1.1

**Costruzione del grafo**: Si usa il kernel gaussiano $s(x,y)=\exp{-(\lVert x-y\rVert_2^2 / (2\sigma)^2)}$ con $\sigma=1.7$

![center](./presentation_images/results_1.png)

---

# Esperimento 1.2

**Costruzione del grafo**: KNN-embedding. Per lo SC classico, $k=60$; per lo SC basato su SVD, $k=15$.

![width:900px center](./presentation_images/results_1_knn.png)

---

# Esperimento 2

**Costruzione del Grafo**: Per i due cerchi abbiamo usato la stessa dell'Esperimento 1. Per il dataset delle mezze lune, abbiamo utilizzato il KNN-graph per rappresentare il grafo 

**Parametri**: Per il dataset dei due cerchi abbiamo variato l'iperparametro $l$ con $l=1$, $l=2$. Per il dataset delle due mezze lune, $k = 5, 15$

---

# Esperimento 2 (Risultati)

![center](./presentation_images/results_circles.png)

---

# Esperimento 2 (Risultati)

![center](./presentation_images/results_moons.png)

---

# Esperimento 3: deduzione dei parametri

* Per la costruzione del grafo, useremo il kernel gaussiano parametrizzato in $\sigma > 0$:
$$
s(x,y)=\exp(- \lVert x-y \rVert_2^2 / 2\sigma^2)
$$

* Supponiamo di non conoscere nè $k$ nè $\sigma$: come possiamo determinare tali parametri?

* Le seguenti formule ci potranno *"suggerire"* tali parametri. Dati $(\sigma_i)_i$ tratti da $\Sigma$,

$$
\begin{gather}
(1) \  \sigma_k - \sigma_{k+1} \\
(2) \ \sigma_k - 2\sigma_{k+1} + \sigma_{k+2} \\
(3)\ \frac{\sum_{i \leq k} \sigma_i}{ \sum_{i \leq N} \sigma_i} \geq \theta
\end{gather}
$$


---

# Esperimento 3: analisi

![center](./presentation_images/parameters_1.png)

---

# Esperimento 3: analisi

![center](./presentation_images/parameters_2.png)

---

# Esperimento 3: risultati

Abbiamo quindi dedotto le variabili $k=3$ e $\sigma=2$

![center](./presentation_images/results_3.png)
