# Filament Review Analysis
[Main Report](https://github.com/nhansendev/FilamentReviewAnalysis/blob/main/FilamentReviewAnalysis.pdf)

[Presentation](https://github.com/nhansendev/FilamentReviewAnalysis/blob/main/FilamentReviewAnalysis_Presentation.pdf)

Tools/Techniques Used:
- python
- matplotlib, pandas
- Jupyter Notebooks
- Natural Language Processing
- TF-IDF
- BERTopic (Sentence Transformers)
- Supervised Learning
- Clustering Algorithms

### Abstract

In this project topic modeling is used to extract actionable insights from product reviews for 3D printer filament. Using this information the factors important to customers when purchasing 3D printer filament can be estimated, as well as more specific feedback on a case-by-case basis, such as per supplier, or filament type. Reviews were retrieved from the AMAZON REVIEWS 2023 dataset after careful filtering was performed to identify relevant products, which required the use of supervised classification algorithms. Topic modeling was performed using the BERTopic model to extract common discussion topics, from which actionable insights could be drawn. Topic comparisons were performed using a variety of metrics, including the frequencies at which topics were paired within reviews, and topic tones. These comparisons revealed several useful insights into customer preferences and common complaints, which could be expanded upon further in future analysis.

### Insights:

From the summaries and topic comparisons there are several insights and suggestions that can be extracted:
- One of the most significant properties for customers is the appearance of the filament, specifically the color. Judging the color of the filament from online photos can often be misleading, which might be improved by using a variety of lighting conditions, print samples, etc.
- Spools becoming tangled, or jammed significantly frustrates customers, requiring careful control of the winding process.
- Standardizing spool dimensions and offering recyclable cardboard spools may improve customer experiences by allowing the use of one-size-fits-all spool holders and empty spools to be easily recycled. Cardboard may provide additional cost savings over custom injection-molded spools.
- Filament that has absorbed moisture tends to have degraded qualities, but absorption can be minimized through proper packaging in vacuum bags with desiccant.
- There may be opportunities to provide accessories with the filament, such as specialized tape and glue to improve bed adhesion, filament dryers, desiccant and dry-boxes to remove and prevent moisture accumulation, and filament spool holders.
- Print settings frequently require experimentation to dial-in, so providing a set of recommended settings to start with may improve user experience. 
- Customers seem to prefer one-size-fits-all solutions where they can acquire all their filament from one supplier, but will switch to other brands if filament availability or quality becomes inconsistent. Improving these metrics may improve customer retention.
