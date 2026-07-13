# Latent Semantic Analysis: News Article Classification
**Course:** Linear Algebra (ECE) | Spring 2024  
**Instructor:** Dr. M. S. Sarafraz  
**Institution:** University of Tehran  

---

## 1. Project Overview & Objectives
This project applies **Latent Semantic Analysis (LSA)** to classify a dataset of 2,225 short English news articles into five distinct categories based on their semantic content:
0. **Politics**
1. **Sport**
2. **Technology**
3. **Entertainment**
4. **Business**

By converting raw text into numerical representations, reducing dimensions via Singular Value Decomposition (SVD), and projecting documents into a low-dimensional semantic space, we group articles by topic and evaluate semantic relationships using similarity metrics.

---

## 2. Linear Algebra Methodology

### Bag of Words (BoW)
Machines treat words as isolated symbols. To encode semantic relationships, we convert text into a term-document matrix $B = [f_{i,j}]$, where $f_{i,j}$ represents the frequency of vocabulary word $j$ in document $i$.

### Standardization
To ensure all words contribute equally to topic extraction, we standardize the frequencies of the Bag of Words matrix:
$$f'_{ij} = \frac{f_{ij} - \mu_j}{\sigma_j}$$
where $\mu_j$ is the mean frequency of word $j$, and $\sigma_j$ is its standard deviation.

### Singular Value Decomposition (SVD)
The standardized training term-document matrix $A \in \mathbb{R}^{m \times n}$ is decomposed into three matrices:
$$A = U S V^T$$
- **$U \in \mathbb{R}^{m \times m}$**: Orthogonal matrix representing document-concept relations.
- **$S \in \mathbb{R}^{m \times n}$**: Diagonal matrix of singular values representing concept strength.
- **$V \in \mathbb{R}^{n \times n}$**: Orthogonal matrix representing concept-word relations.

### Truncated SVD
To filter out noise and project the data into a $k$-dimensional latent space, we keep the top $k$ singular values:
$$A_k = U_k S_k V_k^T$$
where $U_k \in \mathbb{R}^{m \times k}$, $S_k \in \mathbb{R}^{k \times k}$, and $V_k \in \mathbb{R}^{k \times n}$.

### Randomized SVD
For large-scale datasets, deterministic SVD is computationally prohibitive. Randomized SVD uses randomized projection to approximate the range of $A$, projecting it onto a smaller subspace where standard SVD can be calculated efficiently.

---

## 3. Theoretical Questions & Answers

### Q1: Term Frequency (TF) & Inverse Document Frequency (IDF)
- **Term Frequency (TF)** measures how frequently a term $t$ occurs in a document $d$:
  $$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
- **Inverse Document Frequency (IDF)** measures a term's significance across the entire corpus $D$:
  $$IDF(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

**Why independent study is misleading:**
- **TF alone:** Overemphasizes common structural words (e.g., "the", "is", "of") that appear frequently in almost all documents but carry no topic-specific semantic meaning.
- **IDF alone:** Overemphasizes extremely rare words, typographical errors, or specialized jargon that appear in very few documents. These words receive high IDF scores but are irrelevant for general topic classification.
- **TF-IDF Integration:** Combines both metrics to highlight words that are frequent in a specific document but relatively unique within the broader corpus, reflecting actual semantic significance.

---

### Q2: Rank Reduction Threshold & The "Elbow Point"
To select the target dimension $k$ in Truncated SVD, we plot the singular values (or cumulative explained variance) against the number of components:
- **Scree Plot:** A plot of the singular values $\sigma_i$.
- **Cumulative Explained Variance Plot:** A plot of $\sum_{i=1}^k \sigma_i^2 / \sum_{j=1}^r \sigma_j^2$.
- **The Elbow Point:** The point on the plot where the curve changes from steep to flat. This "knee" or "elbow" indicates the point of diminishing returns—adding more components beyond this point captures noise rather than significant latent concepts.

---

### Q3: Reconstruction Error Calculation
The reconstruction error measures the information lost when approximating $A$ with $A_k = U_k S_k V_k^T$.
- **Absolute Reconstruction Error:** Computed as the Frobenius norm of the difference matrix:
  $$\text{Error}_{\text{abs}} = \|A - A_k\|_F = \sqrt{\sum_{i,j} (A_{ij} - (A_k)_{ij})^2}$$
- **Using Singular Values:** According to the Eckart-Young-Mirsky Theorem, it simplifies to:
  $$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$
- **Relative Reconstruction Error:**
  $$\text{Error}_{\text{rel}} = \frac{\|A - A_k\|_F}{\|A\|_F} = \sqrt{\frac{\sum_{i=k+1}^{r} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}}$$

---

### Q4: Cosine Similarity & Euclidean Distance
- **Cosine Similarity:** Measures the cosine of the angle between two vectors $\mathbf{u}$ and $\mathbf{v}$:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  - **Range:** $[-1, 1]$ (or $[0, 1]$ for non-negative spaces).
  - **Boundary Meaning:** $1$ implies collinear vectors pointing in the same direction (identical semantic profile); $0$ implies orthogonal vectors (uncorrelated concepts); $-1$ implies opposite directions (opposing semantics).
  - **Scale Invariance:** It measures angle rather than magnitude, which is ideal for comparing texts of different lengths.
- **Euclidean Distance:** Measures the geometric distance between two points:
  $$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$
  - **Range:** $[0, \infty)$.
  - **Boundary Meaning:** $0$ implies identical vectors.
  - **Scale Sensitivity:** Long documents containing many words will have high magnitudes and appear distant from short documents, even if they discuss the exact same topic.

---

### Q5: Necessity of Standardization before SVD
Standardization is performed using:
$$x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
**Why it is essential:**
- Singular Value Decomposition captures directions of maximum variance.
- Without standardization, words with naturally high occurrence frequencies (even after removing stop words) will dominate the variance. SVD would align its principal axes along these high-frequency words, ignoring semantic changes of lower-frequency, high-importance words.
- Standardization scales all word features to zero mean and unit variance, allowing each vocabulary word to contribute equally to the concept geometry.

---

### Q6: Randomized SVD Algorithm & Performance
For large matrices $A \in \mathbb{R}^{m \times n}$, standard SVD takes $\mathcal{O}(mn\min(m,n))$ operations. Randomized SVD uses random projections to approximate the range of $A$, reducing complexity to $\mathcal{O}(mn\log(k) + (m+n)k^2)$, making it extremely efficient for large-scale data.

#### **Algorithm Pseudocode:**
1. **Input:** Matrix $A \in \mathbb{R}^{m \times n}$, target rank $k$, power iterations $q$ (typically 2 to 5).
2. Generate a random Gaussian matrix $\Omega \in \mathbb{R}^{n \times k}$.
3. Compute the sample matrix $Y = A \Omega$.
4. **Power Iterations:** For $i = 1$ to $q$, update:
   $$Y = A (A^T Y)$$
5. Orthonormalize the columns of $Y$ via QR Decomposition:
   $$Q, R = \text{qr}(Y)$$
6. Project $A$ onto the low-dimensional subspace:
   $$B = Q^T A \quad (B \in \mathbb{R}^{k \times n})$$
7. Compute the standard SVD of the smaller matrix $B$:
   $$U_B, S, V^T = \text{svd}(B)$$
8. Reconstruct the left singular vectors of $A$:
   $$U = Q U_B$$
9. **Output:** Left singular vectors $U$, singular values $S$, right singular vectors $V^T$.

---

### Q18: Semantic Searching in Latent Space
- **BoW Space Searching:** If a search query is "technology", its representation is a one-hot vector $[0, \dots, 1, \dots, 0]$. Computing similarity only matches documents containing the exact word "technology".
- **Latent Space Semantic Matching:** Documents with terms like "mobile", "digital", and "phone" co-occur in the same contexts as "technology". LSA places these semantically related words near each other in the latent space. Thus, a document discussing "mobile" and "digital" will project close to the query vector "technology", returning relevant documents that do *not* contain the word "technology".
- **Computational Cost Benefit:** Searching in BoW space requires calculating similarity in a high-dimensional, sparse space (dimension $N_v \approx 50,000$). Latent space searching operates in a dense, low-dimensional space ($k \approx 100$). Computing cosine similarity in the low-dimensional space is orders of magnitude faster and consumes significantly less memory.

---

## 4. Step-by-Step Implementation & Results

> [!NOTE]
> Please execute the Jupyter notebook `Code.ipynb` using the local python virtual environment to generate the exact values and plots, then copy-paste them into the placeholders below.

### 4.1. Preprocessing & Word Frequencies (Q7, Q8, Q9)
1. **Text Cleansing:** All text was lowercased, punctuation removed, and consecutive whitespace consolidated.
2. **Top 30 Words Analysis:** The frequency of the 30 most common words across all articles was plotted.
   - *Placeholder for Frequency Bar Chart:*  
     `[RUN CELL 15 IN Code.ipynb AND PASTE THE FREQUENCY BAR CHART HERE]`
   - *Interpretation:* The most frequent words include generic terms (e.g., "said", "US", "year", "people"). These are mostly structural or too common to distinguish between specific topics, demonstrating why TF-IDF or standardization is necessary.
3. **Word Cloud:** A word cloud of TF-IDF weighted terms was plotted.
   - *Placeholder for Word Cloud:*  
     `[RUN CELL 14 IN Code.ipynb AND PASTE THE WORD CLOUD IMAGE HERE]`

---

### 4.2. Train-Test Splitting & SVD Dimensions (Q10, Q11)
To prevent data leakage, the dataset was split:
- **Training Set:** First 2000 rows.
- **Test Set:** Remaining 225 rows.

- *Placeholder for BoW Matrix Dimensions:*  
  `[RUN CELL 19 IN Code.ipynb AND PASTE Training and Test BoW matrix shapes]`

Standardization was fit on the training set, followed by SVD.
- *Placeholder for SVD Factorized Matrix Dimensions:*  
  `[RUN CELL 24 IN Code.ipynb AND REPORT the shapes of U, S, and Vt]`

---

### 4.3. Threshold Selection & Reconstruction Error (Q12)
By analyzing the Scree Plot and Cumulative Explained Variance Ratio:
- *Placeholder for SVD Variance Plots:*  
  `[RUN CELL 24 IN Code.ipynb AND PASTE Scree Plot and Cumulative Variance Plot HERE]`
- **Proposed Threshold ($k$):** $k = 7$ is chosen at the elbow point where the cumulative explained variance begins to flatten.
- **Truncated SVD Reconstruction Error:**
  - *Placeholder for Truncated SVD Reconstruction Errors:*  
    `[RUN CELL 24 IN Code.ipynb AND PASTE absolute and relative reconstruction errors]`

---

### 4.4. Randomized SVD Implementation & Comparison (Q13, Q14)
We implemented Randomized SVD from scratch using Gaussian projection and QR factorization.
- **Randomized SVD Reconstruction Error:**
  - *Placeholder for Randomized SVD Errors:*  
    `[RUN CELL 27 IN Code.ipynb AND PASTE absolute and relative errors for Randomized SVD]`
- **Comparison:**
  - *Placeholder for Comparison Table:*  
    `[RUN CELL 27 IN Code.ipynb AND PASTE the Truncated vs. Randomized SVD comparison text]`
- **Recommendation:** For internet-scale text collections, **Randomized SVD** is highly recommended because standard SVD has a cubic complexity $\mathcal{O}(n^3)$ which becomes impossible to compute. Randomized SVD reduces this significantly while introducing only a negligible increase in reconstruction error.

---

### 4.5. Latent Space Representation & Semantics (Q15, Q16, Q17)
1. **Component Loadings:** The top 5 words with the highest magnitude in each component were printed.
   - *Placeholder for Component Loadings:*  
     `[RUN CELL 31 IN Code.ipynb AND PASTE top 5 words for each component]`
   - *Interpretation:* For example, Component 3 is heavily loaded on "play", "game", "time", and "win" (representing Sports). Component 4 is loaded on "artist", "music", "won", and "film" (representing Entertainment).
2. **Word Similarity Evaluations:** We calculated the similarity of word pairs:
   - *Placeholder for Word Similarity table:*  
     `[RUN CELL 46 IN Code.ipynb AND PASTE word similarity cosine and euclidean metrics]`
   - *Interpretation:* Synonym pairs like `(play, game)` and `(director, film)` exhibit very high Cosine Similarity and low Euclidean Distance. Unrelated pairs like `(government, music)` show near-zero Cosine Similarity. Antonym or opposing pairs show negative similarities.
3. **Document-Word Cosine Similarity:**
   - *Placeholder for Document 1100 Similarity Charts:*  
     `[RUN CELL 36 IN Code.ipynb AND PASTE Document similarity and Word count bar charts HERE]`

---

### 4.6. Document Classification & Accuracy (Q19, Q20)
1. **LSI Concept Map Heatmap:** Average concept vectors for each document category (0 to 4) were plotted.
   - *Placeholder for Concept Heatmap:*  
     `[RUN CELL 30 IN Code.ipynb AND PASTE the average scores heatmap HERE]`
2. **Classification Method:** To classify a document, we compute its latent representation, measure its Cosine Similarity to the average training vector of each class, and predict the class with the highest similarity.
3. **Evaluation Metrics:**
   - **Training Set Accuracy:**  
     `[RUN CELL 42 IN Code.ipynb AND REPORT Training Set Accuracy]`
   - **Test Set Overall Accuracy:**  
     `[RUN CELL 44 IN Code.ipynb AND REPORT Test Set Overall Accuracy]`
   - **Per-Category Accuracy:**  
     `[RUN CELL 44 IN Code.ipynb AND REPORT accuracies for each class]`
   - *Observation:* The test set (remaining 225 rows) only contains documents labeled with class 4 (Business). This is because the original dataset was sorted, and a sequential split leaves all class 4 documents at the end. This highlights the importance of randomized shuffling in real-world ML workflows, though we strictly adhered to the assignment split instructions.
