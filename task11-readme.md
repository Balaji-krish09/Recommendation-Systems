# Machine Learning Task — 11: Recommendation Systems Foundations & Content-Based Filtering

> **Goal:** Learn how to frame a recommendation problem correctly — queries, items, the feedback matrix, explicit vs. implicit signals — then build your first working recommender using content-based filtering: representing items as feature vectors and recommending by similarity.

> **Note:** This opens a new track — Recommendation Systems, end-to-end. Task 11 is the foundation every later task builds on: the vocabulary, the feedback matrix, and the simplest recommender family (content-based), which needs no other users' data at all.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What You Will Learn](#2-what-you-will-learn)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Phase 1 — What Is a Recommendation System?](#phase-1--what-is-a-recommendation-system)
5. [Phase 2 — The Feedback Matrix: Explicit vs. Implicit Signals](#phase-2--the-feedback-matrix-explicit-vs-implicit-signals)
6. [Phase 3 — Content-Based Filtering: Core Idea & Item Profiles](#phase-3--content-based-filtering-core-idea--item-profiles)
7. [Phase 4 — Representing Text Items: TF-IDF](#phase-4--representing-text-items-tf-idf)
8. [Phase 5 — Cosine Similarity & Ranking Items](#phase-5--cosine-similarity--ranking-items)
9. [Phase 6 — Building User Profiles from Item Features](#phase-6--building-user-profiles-from-item-features)
10. [Phase 7 — Strengths & Weaknesses of Content-Based Filtering](#phase-7--strengths--weaknesses-of-content-based-filtering)
11. [Phase 8 — Capstone: An End-to-End Content-Based Movie Recommender](#phase-8--capstone-an-end-to-end-content-based-movie-recommender)
12. [Exercises (Must Do)](#exercises-must-do)
13. [Deliverables Checklist](#deliverables-checklist)
14. [Reference: Key Concepts Glossary](#reference-key-concepts-glossary)
15. [Resources](#resources)

---

## 1. Project Overview

### The Scenario

Every streaming service, e-commerce site, and news app you use runs a recommendation system behind the scenes. Task 11 starts where every real recsys project starts: correctly framing the problem (who is the "query," what is being recommended, what counts as a positive signal) before writing a line of modeling code. Then you'll build the simplest recommender family — content-based filtering — which recommends items similar to what a user already liked, using nothing but the items' own features.

### Why This Matters

| Concept | Where It Shows Up in Practice |
|---|---|
| Query / item framing | Determines your entire data pipeline — get this wrong and nothing downstream works |
| Feedback matrix | The core data structure every recommender algorithm (Tasks 11–15) is built on |
| Explicit vs. implicit feedback | Determines which loss functions and metrics are even valid later |
| Content-based filtering | The "zero cold-start-for-new-items" baseline every recsys should have |
| TF-IDF + cosine similarity | A technique you'll reuse constantly, in recsys and far beyond (search, dedup, clustering) |

---

## 2. What You Will Learn

By completing this task, you will be able to:

- Define a recommendation system's two core entities: the **query** (the context — a user, their history, time, device) and the **item** (the thing being recommended)
- Classify a recommendation algorithm as content-based, collaborative, or hybrid, and explain what data each requires
- Build and read a **feedback matrix** `A ∈ ℝ^(m×n)` where rows are users/queries and columns are items
- Distinguish explicit feedback (ratings, thumbs up/down) from implicit feedback (clicks, watches, purchases), and explain why implicit feedback dominates real production systems
- Explain content-based filtering's core idea: recommend items similar to items the user has already liked, using item features alone
- Build an item profile from categorical, numeric, and free-text features
- Compute TF-IDF vectors for text-based item descriptions and explain what each term (term frequency, inverse document frequency) is correcting for
- Compute cosine similarity between item vectors and between a user profile and candidate items
- Build a user profile as a weighted average of the item vectors the user has liked
- List content-based filtering's core limitations: no serendipity (never recommends outside what you already like), requires good item features, and the "filter bubble" effect
- Build a complete content-based recommender end-to-end on a real dataset

---

## 3. Prerequisites & Setup

### Required Knowledge

- Task 3 (vectors, dot products — the TF-IDF and cosine similarity math builds directly on this)
- Basic Pandas/NumPy

### Environment Setup

```bash
python3 -c "import numpy, pandas, sklearn"
```

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### Dataset

Use the [MovieLens 100K or 25M dataset](https://grouplens.org/datasets/movielens/) (free, standard benchmark for recsys coursework) for this entire task series. It ships `movies.csv` (`movieId, title, genres`), `ratings.csv` (`userId, movieId, rating, timestamp`), and (in the full version) `tags.csv`.

```python
movies = pd.read_csv("movies.csv")     # movieId, title, genres
ratings = pd.read_csv("ratings.csv")   # userId, movieId, rating, timestamp
```

---

## Phase 1 — What Is a Recommendation System?


### Tasks to Perform

1. Define a recommendation system in one sentence: a system that predicts, for a given query, which items a user is most likely to engage with, out of a (often huge) catalog.
2. Define "query" (also called "context"): can be a user ID alone, or a richer combination of user features, the item they're currently viewing, time of day, device, and location.
3. Define "item": the entity being recommended — movies, products, videos, jobs, songs, news articles.
4. List at least 4 industries and name the query/item pair for each (e.g., Netflix: query = user + watch history, item = movie/show; LinkedIn: query = user profile, item = job posting).
5. Classify these three algorithm families and state what data each needs: **content-based filtering** (item features only), **collaborative filtering** (the user-item interaction matrix only, no item features needed), **hybrid** (both).
6. Explain why recommendation is fundamentally different from search: search starts from an explicit query typed by the user; recommendation often has to guess relevant items with little or no explicit query at all ("the zero-query problem").

**Deliverable for Phase 1:** A table of 4 real-world recommender examples (platform, query, item, algorithm family likely used), and a one-paragraph explanation of the "zero-query problem."

---

## Phase 2 — The Feedback Matrix: Explicit vs. Implicit Signals



### Tasks to Perform

1. Build the feedback matrix `A` from the MovieLens `ratings.csv`: pivot so rows = `userId`, columns = `movieId`, values = `rating`.
2. Confirm the matrix is overwhelmingly sparse (most user-movie cells are empty) and compute the exact sparsity percentage (`1 - nnz / (m*n)`).
3. Define explicit feedback (a rating, a thumbs up/down, a star review) and implicit feedback (a click, a watch, a purchase, time spent, a skip).
4. State the key tradeoff: explicit feedback is a direct, unambiguous signal but is sparse and has selection bias (users only rate items they chose to engage with); implicit feedback is abundant but noisy (a click doesn't mean "liked," and "no click" doesn't mean "disliked").
5. Convert the explicit `ratings.csv` into an implicit binary matrix (`1` if `rating ≥ 4`, else unobserved) and recompute sparsity — note how differently the two framings treat the same raw data.
6. State why most production recommenders (YouTube, Spotify, Amazon) are trained primarily on implicit feedback despite its noise: it's orders of magnitude more abundant than explicit ratings.

### 2.1 Code Reference

```python
feedback_matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
m, n = feedback_matrix.shape
nnz = ratings.shape[0]
sparsity = 1 - nnz / (m * n)
print(f"{sparsity:.4%} sparse")   # typically >98% sparse even on MovieLens 100K
```

**Deliverable for Phase 2:** The pivoted feedback matrix, its sparsity percentage, the implicit binary version, and your written explanation of the explicit-vs-implicit tradeoff.

---

## Phase 3 — Content-Based Filtering: Core Idea & Item Profiles



### Tasks to Perform

1. Explain content-based filtering's core idea in one sentence: recommend items whose **features** resemble the features of items this specific user already liked — no other users' data involved.
2. Build an item profile for each movie from `movies.csv`: parse the pipe-separated `genres` column into a multi-hot vector (one column per genre, 1 if present).
3. Add at least one numeric feature if available (e.g., release year extracted from the title) and discuss whether/how it should be scaled before combining with the multi-hot genre vector.
4. Pick one movie and print its full feature vector, confirming it correctly reflects its genres.
5. Explain, in one sentence, why content-based filtering needs **no other users** at all to make a recommendation for a brand-new item (it only needs that item's own features) — this is the key property that solves the item cold-start problem.

### 3.1 Code Reference

```python
genres_dummies = movies["genres"].str.get_dummies(sep="|")
item_profiles = pd.concat([movies[["movieId", "title"]], genres_dummies], axis=1)
print(item_profiles.head())
```

**Deliverable for Phase 3:** The full multi-hot item-profile table for all movies, and the single-movie feature vector printout with a written explanation.

---

## Phase 4 — Representing Text Items: TF-IDF



### Tasks to Perform

1. Explain why raw word counts are a poor item representation: common words ("the," "movie," "story") dominate the vector without carrying meaningful signal.
2. Write the TF-IDF formula conceptually: `TF-IDF(t, d) = TF(t, d) × IDF(t)`, where `TF(t,d)` is how often term `t` appears in document `d`, and `IDF(t) = log(N / df(t))` downweights terms that appear in many documents.
3. State what each factor corrects for: TF rewards terms central to *this* item's description; IDF punishes terms so common they don't distinguish one item from another.
4. Fit `TfidfVectorizer` on a text field (movie genres treated as a "document," or plot summaries/tags if using the full MovieLens dataset with `tags.csv`).
5. Inspect the vocabulary (`vectorizer.get_feature_names_out()`) and the TF-IDF vector for one item, and identify which terms got the highest weight — confirm they're genuinely distinguishing terms, not generic filler words.

### 4.1 Formula & Code Reference

```
TF-IDF(t, d) = TF(t, d) × log(N / df(t))
   TF(t,d)  = count of term t in document d
   N        = total number of documents (items)
   df(t)    = number of documents containing term t
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Using genres as pseudo-text; swap in real plot summaries/tags if available
movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres_text"])
print(tfidf_matrix.shape)   # (n_movies, vocabulary_size)
```

**Deliverable for Phase 4:** The TF-IDF matrix shape, the vocabulary size, and the top-weighted terms for 2–3 sample items with a written interpretation.

---

## Phase 5 — Cosine Similarity & Ranking Items



### Tasks to Perform

1. Write the cosine similarity formula: `cos(A,B) = (A·B) / (‖A‖·‖B‖)`, and explain why it measures the *angle* between two vectors rather than their magnitude.
2. Explain why cosine similarity is preferred over raw dot product for TF-IDF vectors (it normalizes away the effect of longer/shorter item descriptions).
3. Compute the pairwise cosine similarity matrix across all items using `sklearn.metrics.pairwise.cosine_similarity`.
4. Pick one movie a user liked, retrieve the top-10 most similar movies by cosine similarity, and sanity-check the results (do the recommended genres make sense?).
5. State the "item-item" recommendation rule this implements: "users who liked X also tend to like things similar to X" — note this uses only item features, not any other user's behavior.

### 5.1 Formula & Code Reference

```
cosine_similarity(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

```python
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(tfidf_matrix)

def recommend_similar(movie_title, top_n=10):
    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies.iloc[[i for i, _ in scores]][["title", "genres"]]

print(recommend_similar("Toy Story (1995)"))
```

**Deliverable for Phase 5:** The top-10 similar-items output for at least 2 seed movies, with a written sanity check of the results.

---

## Phase 6 — Building User Profiles from Item Features



### Tasks to Perform

1. Explain the user-profile approach: represent a user as the (weighted) average of the feature vectors of every item they've rated highly.
2. For one user in `ratings.csv`, retrieve every movie they rated `≥ 4`, look up each movie's TF-IDF (or multi-hot genre) vector, and compute the user's profile as the mean of those vectors.
3. Weight the average by rating value instead of a plain mean (e.g., a 5-star movie contributes more than a 4-star movie) and compare the resulting profile to the unweighted version.
4. Compute cosine similarity between this user's profile and every *unrated* movie's vector, and recommend the top-10 highest-similarity unrated movies.
5. Repeat for a second user with clearly different taste (e.g., mostly comedies vs. mostly action) and confirm the two users get visibly different recommendations.

### 6.1 Code Reference

```python
def build_user_profile(user_id, ratings, item_vectors, movie_index):
    user_ratings = ratings[(ratings.userId == user_id) & (ratings.rating >= 4)]
    idxs = [movie_index[mid] for mid in user_ratings.movieId if mid in movie_index]
    weights = user_ratings.rating.values[:len(idxs)]
    if not idxs:
        return None
    vectors = item_vectors[idxs]
    return np.average(vectors.toarray(), axis=0, weights=weights)
```

**Deliverable for Phase 6:** Two distinct user profiles with their top-10 recommendations each, demonstrating personalization.

---

## Phase 7 — Strengths & Weaknesses of Content-Based Filtering



### Tasks to Perform

1. List content-based filtering's strengths: no cold-start problem for new *items* (a brand-new movie can be recommended immediately from its own features), no need for other users' data, recommendations are explainable ("because you liked X").
2. List its weaknesses: limited serendipity (can only ever recommend things similar to what's already liked — never discovers a genuinely new taste), quality is capped by how good your item features are, and it still has a cold-start problem for brand-new *users* with no rating history.
3. Explain the "filter bubble" critique in your own words: a purely content-based system can trap a user in an ever-narrowing recommendation loop.
4. State the natural next step (previewed, not built yet): collaborative filtering, which recommends based on what *similar users* liked rather than item features alone — covered in Task 12.

**Deliverable for Phase 7:** A strengths/weaknesses table and a one-paragraph explanation of the filter bubble effect.

---

## Phase 8 — Capstone: An End-to-End Content-Based Movie Recommender



### Tasks to Perform

1. Combine Phases 3–6 into a single, reusable pipeline: load data → build item profiles (genres + any text features) → build the similarity matrix → build a user profile on demand → return top-N recommendations.
2. Wrap it in a function `recommend_for_user(user_id, top_n=10)` that returns a DataFrame of recommended titles, excluding movies the user has already rated.
3. Test it on 3 different users and manually evaluate whether the recommendations feel reasonable given each user's rating history.
4. Add a simple "explain this recommendation" feature: for each recommended movie, print which of the user's previously-liked movies it's most similar to.
5. Time the full pipeline on the whole dataset and note where it would become a bottleneck at real-world scale (e.g., recomputing user profiles and similarity on every request vs. precomputing).

**Deliverable for Phase 8:** The complete, working `recommend_for_user()` function, tested on 3 users with results and explanations, plus a written note on scaling bottlenecks.

---

## Exercises (Must Do)

| # | Exercise | Skills Practiced |
|---|----------|--------------------|
| 1 | Add release-year as a numeric feature (scaled) to the item profile and check if recommendations change meaningfully | Feature engineering |
| 2 | Compare cosine similarity vs. plain Euclidean distance for ranking items — do they ever disagree on the top result? | Similarity-measure comparison |
| 3 | Build a "more like this" feature for a single movie (no user needed) — this is the purest form of item-item content-based filtering | Item-item recommendation |
| 4 | Simulate the new-user cold-start problem: for a user with zero ratings, what would you recommend, and why can't content-based filtering alone solve it? | Cold-start reasoning |
| 5 | Measure how much the recommendation list changes if you drop the IDF term and use raw TF only | TF-IDF ablation |

---

## Deliverables Checklist

- [ ] **Phase 1:** Real-world query/item table + zero-query-problem explanation
- [ ] **Phase 2:** Feedback matrix, sparsity calculation, explicit-vs-implicit write-up
- [ ] **Phase 3:** Full item-profile table + single-item feature vector explanation
- [ ] **Phase 4:** TF-IDF matrix + top-weighted terms interpretation
- [ ] **Phase 5:** Top-10 similar-items results for 2+ seed movies
- [ ] **Phase 6:** Two distinct user profiles with differing top-10 recommendations
- [ ] **Phase 7:** Strengths/weaknesses table + filter bubble write-up
- [ ] **Phase 8:** Working end-to-end `recommend_for_user()` function with explanations
- [ ] **Exercises:** All 5 exercises attempted
- [ ] **Code:** Clean notebook that runs top to bottom without errors

---

## Reference: Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Query (Context)** | The input to a recommender — a user, their history, and/or situational context |
| **Item** | The entity being recommended |
| **Feedback Matrix** | The `users × items` matrix of observed interactions/ratings |
| **Explicit Feedback** | A direct preference signal: a rating, a thumbs up/down |
| **Implicit Feedback** | An indirect signal inferred from behavior: a click, a watch, a purchase |
| **Content-Based Filtering** | Recommending items similar to items a user already liked, using item features alone |
| **Item Profile** | A feature vector representing an item (categorical, numeric, and/or text-derived features) |
| **User Profile** | A feature vector representing a user, built from the items they've engaged with |
| **TF-IDF** | Term Frequency–Inverse Document Frequency; a text-vectorization weighting scheme that downweights common words |
| **Cosine Similarity** | `(A·B)/(‖A‖‖B‖)`; measures the angle between two vectors, ignoring magnitude |
| **Filter Bubble** | The risk that a purely content-based system narrows a user's exposure over time |
| **Item Cold Start** | The problem of recommending a brand-new item with no interaction history — content-based filtering solves this natively |
| **User Cold Start** | The problem of recommending to a brand-new user with no history — not solved by content-based filtering alone |

---

## Resources

| Resource | Link |
|----------|------|
| MovieLens datasets (GroupLens) | [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/) |
| Scikit-Learn `TfidfVectorizer` documentation | [scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) |

---

## Suggested Timeline

| Date | Phases |
|------|--------|
| 07/10/2026 | Phase 1–8, Exercises, review, Demo|

---
**NOTE:** This begins a  Recommendation Systems track. Keep your notebook and dataset consistent across all five tasks — later tasks build directly on this one.