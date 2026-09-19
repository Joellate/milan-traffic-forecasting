# References

1. G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi, F. Antonelli,
   A. Vespignani, A. Pentland, B. Lepri. "A multi-source dataset of urban life in the city
   of Milan and the Province of Trentino." *Scientific Data* 2, 150055 (2015).
   https://doi.org/10.1038/sdata.2015.55

2. C. Zhang and P. Patras. "Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal
   Neural Networks." *ACM MobiHoc* 2018. arXiv:1712.08083. Uses the same Milan Telecom Italia
   grid dataset; shows ARIMA and Holt-Winters exponential smoothing baselines degrade sharply
   as the forecast horizon grows and traffic fluctuates, motivating deep spatio-temporal models.

3. S. Jaffry. "Cellular Traffic Prediction with Recurrent Neural Network." arXiv:2003.02807
   (2020). Compares ARIMA, feed-forward NN, and LSTM one-step-ahead forecasts on the same
   Milan dataset; finds ARIMA a "decent" but limited baseline and LSTM both accurate and
   faster to train than FFNN.

4. O. Aouedi, V. A. Le, K. Piamrat, Y. Ji. "Deep Learning on Network Traffic Prediction:
   Recent Advances, Analysis, and Future Directions." *ACM Computing Surveys*, 57(6), 2025.
   Broad survey of RNN/LSTM/GRU, CNN, Transformer, and graph-based approaches to network
   (incl. cellular) traffic prediction.

5. A. Vaswani, N. Shazeer, N. Parmar, et al. "Attention Is All You Need." *NeurIPS* 2017.
   Introduces the Transformer / self-attention architecture used for Model 3.

6. H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, W. Zhang. "Informer: Beyond
   Efficient Transformer for Long Sequence Time-Series Forecasting." *AAAI* 2021.
   Establishes Transformer variants as competitive for time-series forecasting.

7. A. Zeng, M. Chen, L. Zhang, Q. Xu. "Are Transformers Effective for Time Series
   Forecasting?" *AAAI* 2023. Critical counterpoint showing simple linear/recurrent models
   can match or beat Transformers on many forecasting benchmarks — used to motivate a
   cautious, evidence-based reading of the Transformer's results in this study.

8. S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory." *Neural Computation*,
   9(8), 1735-1780 (1997). Foundational LSTM reference for Model 2.

9. R. J. Hyndman and G. Athanasopoulos. *Forecasting: Principles and Practice*, 3rd ed.
   OTexts, 2021, ch. 12 (complex seasonality / dynamic harmonic regression). Motivates using
   Fourier terms rather than a large seasonal-order SARIMA for high-frequency data with
   multiple seasonal periods.

10. A. M. De Livera, R. J. Hyndman, R. D. Snyder. "Forecasting Time Series With Complex
    Seasonal Patterns Using Exponential Smoothing." *Journal of the American Statistical
    Association*, 106(496), 1513-1527 (2011). Supports Fourier/harmonic treatment of
    simultaneous daily + weekly seasonality, directly applicable to 10-minute mobile
    traffic data (daily period = 144, weekly period = 1008).

11. Y.-A. de Montjoye, C. A. Hidalgo, M. Michell, A. Clauset. "Unique in the Crowd: The Privacy
    Bounds of Human Mobility." *Scientific Reports* 3, 1376 (2013).
    https://doi.org/10.1038/srep01376

Dataset access:
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU
