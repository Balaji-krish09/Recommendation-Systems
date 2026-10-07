# Recommendation Systems

A five-task track (Tasks 11–15) building recommender systems end to end on the
[MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/) dataset
(610 users · 9,742 movies · 100,836 ratings · 3,683 tags).

## Task 11: Recommendation-system foundations & content-based filtering

**Notebook:** [`Task11_RecommendationSystems_ContentBased.ipynb`](Task11_RecommendationSystems_ContentBased.ipynb)
**Slides:** [`Task11_RecommendationSystems_Presentation.pptx`](Task11_RecommendationSystems_Presentation.pptx)
**Task spec:** [`task11-readme.md`](task11-readme.md)

| Phase | What was done | Key result |
|---|---|---|
| 1. Framing | Query vs. item, algorithm families, the zero-query problem | 5-platform query/item table |
| 2. Feedback matrix | Users × movies rating matrix, explicit vs. implicit feedback | 98.30% sparse; 99.18% after the implicit (rating ≥ 4) conversion |
| 3. Item profiles | Multi-hot genre vectors + scaled release year | Item-profile table 9,742 × 24 |
| 4. TF-IDF | Genre TF-IDF (fixed tokenizer) + user-tag TF-IDF | 9,742 × 19 genres, 9,742 × 722 tag terms |
| 5. Cosine similarity | Item-item similarity, top-10 similar movies | 9,742 × 9,742 matrix (380 MB, float32); tags break genre ties |
| 6. User profiles | Rating-weighted mean of liked movies' vectors | Comedy fan vs. action fan: 0 overlap in top-10 |
| 7. Trade-offs | Strengths, weaknesses, filter bubble | Comedy fan: 11 liked genres → 1 recommended genre |
| 8. Capstone | `recommend_for_user(user_id, top_n=10)` with "because you liked" explanations | Tested on 3 users; a few ms per request |

Plus all 5 exercises: year feature, cosine vs. Euclidean, "more like this", new-user cold start, TF-IDF ablation.

### Try it
```python
recommend_for_user(164)                         # any userId from 1 to 610
recommender.more_like_this("Toy Story (1995)")  # item-item, no user needed
```

### How to run
1. Install: `pip install numpy pandas scikit-learn scipy jupyter`
2. The `ml-latest-small/` folder (MovieLens data, with its license in `README.txt`) sits next to the notebook.
3. Open the notebook and run all cells top to bottom.

### Fixes made to the task's reference code
- Genre tokenizer: the default split `Sci-Fi` → `sci` + `fi`; now splits on `|`
- `recommend_similar()`: the seed movie could appear in its own list when scores tie; the seed is now excluded by index
- `build_user_profile()`: `weights[:len(idxs)]` could misalign weights; rows are now filtered first
- Removed the non-content tag "in netflix queue"

## Coming next
- Task 12: Collaborative filtering
- Tasks 13–15: building on the same dataset and notebook

## Data license
MovieLens data © GroupLens Research, University of Minnesota. It is redistributed here under the
same license conditions (see `ml-latest-small/README.txt`), for non-commercial use only.
