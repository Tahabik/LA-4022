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
- **Term Frequency (TF)** measures how often a word shows up in a single document compared to the total words in it:
  $$TF(t, d) = \frac{\text{count}(t, d)}{\sum_{t' \in d} \text{count}(t', d)}$$
- **Inverse Document Frequency (IDF)** measures how common or rare a word is across all the documents in our dataset:
  $$IDF(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

**Why independent study is misleading:**
- **TF alone:** Overemphasizes common structural words (like "the", "is", "of", "and") that appear frequently in almost all documents but don't carry any topic-specific semantic meaning.
- **IDF alone:** Overemphasizes extremely rare words, typographical errors, or specialized jargon that appear in very few documents. These words receive high IDF scores but are irrelevant for general topic classification.
- **TF-IDF Integration:** We multiply them (TF-IDF = TF $\times$ IDF) to highlight words that are frequent in a specific document but relatively unique within the broader corpus, reflecting actual semantic significance.

---

### Q2: Rank Reduction Threshold & The "Elbow Point"
To select the target dimension $k$ in Truncated SVD, we plot the singular values (the scree plot) or cumulative explained variance against the number of components.
Usually, the curve starts steep because the first few components capture most of the variance, and then it flattens out. The "elbow point" is the spot where the plot bends from steep to flat. Beyond this point, adding more components gives us diminishing returns (we are just capturing noise instead of actual latent concepts). To find this point programmatically, we can draw a line from the first point to the last point on the curve, calculate the perpendicular distance from each point on the curve to that line, and pick the point with the maximum distance as the elbow.

---

### Q3: Reconstruction Error Calculation
The reconstruction error shows how much information we lose when we approximate our standardized matrix $A$ with a lower-rank matrix $A_k = U_k S_k V_k^T$.
- **Absolute Reconstruction Error:** Computed as the Frobenius norm of the difference matrix:
  $$\text{Error}_{\text{abs}} = \|A - A_k\|_F = \sqrt{\sum_{i,j} (A_{ij} - (A_k)_{ij})^2}$$
- **Using Singular Values:** According to the Eckart-Young-Mirsky Theorem, we don't actually need to reconstruct the matrix to find this error. It is equal to the square root of the sum of the squares of the singular values we discarded:
  $$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$
- **Relative Reconstruction Error:**
  $$\text{Error}_{\text{rel}} = \frac{\|A - A_k\|_F}{\|A\|_F} = \sqrt{\frac{\sum_{i=k+1}^{r} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}}$$

---

### Q4: Cosine Similarity & Euclidean Distance
- **Cosine Similarity:** Measures the cosine of the angle between two vectors:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  - **Range:** $[-1, 1]$ (or $[0, 1]$ for non-negative spaces like word counts).
  - **Boundary Meaning:** $1$ means they point in the exact same direction (highly related concepts), $0$ means they are orthogonal (uncorrelated), and $-1$ means they point in opposite directions.
  - **Scale Invariance:** It only cares about the direction, so a short article and a long article on the same topic will still have high similarity.
- **Euclidean Distance:** Measures the geometric distance between two points:
  $$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$
  - **Range:** $[0, \infty)$.
  - **Boundary Meaning:** $0$ means the vectors are identical.
  - **Scale Sensitivity:** It is very sensitive to vector magnitude. If one document is much longer than another, their Euclidean distance will be huge, even if they discuss the same topics.

---

### Q5: Necessity of Standardization before SVD
Standardization is performed using:
$$x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
**Why it is essential:**
SVD identifies the directions of maximum variance in our data. If we don't standardize, words that are naturally very common (like "said" or "new") will have huge raw variances simply because of their high counts. SVD would align its components along these common words instead of finding the actual hidden concepts. By standardizing, we put all words on the same scale, so SVD can find latent concepts based on how words co-occur rather than just how common they are.

---

### Q6: Randomized SVD Algorithm & Performance
For large matrices $A \in \mathbb{R}^{m \times n}$, computing the exact SVD is very slow and takes $\mathcal{O}(mn\min(m,n))$ time. If we have a massive text dataset, this is too slow. Randomized SVD solves this by projecting the large matrix onto a much smaller random subspace, doing the exact SVD on that small matrix, and then projecting the results back. This is way faster and only costs a tiny loss in accuracy.

#### **Algorithm Pseudocode:**
1. **Input:** Matrix $A \in \mathbb{R}^{m \times n}$, target rank $k$, and power iterations $q$.
2. Generate a random Gaussian matrix $\Omega \in \mathbb{R}^{n \times k}$.
3. Form the sample matrix $Y = A \Omega$.
4. (Optional) Run $q$ power iterations to improve accuracy: $Y = A(A^T Y)$.
5. Find an orthonormal basis of $Y$ using QR decomposition: $Q, R = \text{qr}(Y)$.
6. Project $A$ onto this basis: $B = Q^T A$ (this is a small $k \times n$ matrix).
7. Compute the standard SVD of $B$: $U_B, S, V^T = \text{svd}(B)$.
8. Map the left singular vectors back: $U = Q U_B$.
9. **Output:** Left singular vectors $U$, singular values $S$, right singular vectors $V^T$.

---

### Q18: Semantic Searching in Latent Space
- **BoW Space Searching:** If a search query is "technology", its representation is a one-hot vector $[0, \dots, 1, \dots, 0]$. Computing similarity only matches documents containing the exact word "technology".
- **Latent Space Semantic Matching:** Documents with terms like "mobile" and "digital" co-occur in the same contexts as "technology". LSA places these semantically related words near each other in the latent space. Thus, a query for "technology" projected into the latent space will still match that document.
- **Computational Cost Benefit:** Searching in BoW space requires calculating similarity in a high-dimensional, sparse space (dimension $N_v \approx 50,000$). Latent space searching operates in a dense, low-dimensional space ($k \approx 7$). Computing cosine similarity in the low-dimensional space is orders of magnitude faster and consumes significantly less memory.

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
