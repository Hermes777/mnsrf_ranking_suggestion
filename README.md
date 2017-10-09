# Multitask Neural Session Relevance Model for Document Ranking and Query Suggestion

<p align="justify">
We propose to use a multitask learning framework to jointly learn document ranking and query suggestion, where improved performance on both tasks can be achieved. We propose a multitask neural session relevance model (M-NSRM) that utilizes previous queries and click-through information from the same search session. M-NSRM is composed of two major components, document ranker and query recommender. Document ranker combines current queryand previous session information and compares the combined representation with the document representation to rank the documents. Query recommender tracks users’ query reformulation sequence considering all previous in-session queries using a sequence to sequence approach. The proposed model is depicted in the following figure.
<p align="justify">
<p align="center">
<br>
<img src="http://i.imgur.com/LnfZrPf.png" width="90%">
<p align="center">

<p align="justify">
Figure: Architecture of the Multitask Neural Session Relevance Model (M-NSRM). M-NSRM uses bi-LSTM with max pooling to form query and document representations and use LSTM to gather session-level information. These recurrent states (current query representation and session-level recurrent state, which summarizes all previous queries)  are used by query decoder and document ranker for predicting next query and computing relevance scores.
<p align="justify">

