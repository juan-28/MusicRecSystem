---

# Music Recommendation System

## Introduction

This project explores music recommendation systems leveraging the Million Song Dataset (MSD) and EchoNest user play history. It focuses on algorithms like Alternating Least Squares (ALS) for matrix factorization and Word2Vec, TF-IDF, and Latent Dirichlet Allocation (LDA) for lyrics-based song recommendations.

## Features

- **Collaborative Filtering**: Using ALS algorithm for matrix factorization based on user listening data.
- **Content-Based Filtering**: Employing TF-IDF, Word2Vec, and LDA for song recommendation based on lyrics data.
- **Scalable Architecture**: Utilization of Amazon EMR clusters and Amazon S3 for handling large datasets.
  
<img width="662" alt="Screenshot 2567-01-30 at 10 50 31" src="https://github.com/juan-28/MusicRecSystem/assets/55826125/d03bd421-3301-4791-882d-994aaaedcd48">

## Dataset

The project utilizes the Million Song Dataset, enriched with additional data like lyrics from musiXmatch, user-generated tags from Last.fm, and listening data from EchoNest.

## Methodology

1. **ALS Model for Collaborative Filtering**: Matrix factorization using the ALS algorithm.
   
<img width="856" alt="Screenshot 2567-01-30 at 10 49 55" src="https://github.com/juan-28/MusicRecSystem/assets/55826125/45c77cc0-51a4-409b-940c-68ea9899b628">

  
3. **Content-Based Filtering Techniques**: 
   - TF-IDF for emphasizing the uniqueness of song lyrics. [Include TF-IDF formula]
   - Word2Vec for capturing contextual relationships in lyrics.
   - LDA for topic modeling and discovering thematic essence in songs.

## Results

- Implementation of ALS and content-based filtering models.
- Analysis of model performance using metrics like Root Mean Square Error (RMSE).
- Insights into thematic patterns and user preferences. [Include screenshot of thematic visualization]

## Limitations and Future Work

Discussion of challenges like the absence of ground truth labels and potential improvements through user feedback integration and hybrid filtering techniques.

## Conclusion

Summarize the effectiveness of the combined approach and the learning outcomes from this project.

## References

List of references used in the project.

---

Please replace the placeholders like "[Include formula for loss function minimization]" and "[Include screenshot of thematic visualization]" with the actual content from your paper where appropriate.
