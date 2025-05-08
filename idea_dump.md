
# Unsorted
- Immunigenicity, glycolycation dataset to get system up and running
- I'm using embeddings from smallest model, might be nice to use 35 mil model, 
- Best to compare to model of similar size.
- Universal approximation theory
- The bigger the model the sweeter the juice
- Adding one layer allows you to reduce model by x
- Could just copy sweetnhet and make it override the import
- don't have too many cells, try to keep many things in one cell.
- Multilabel classification, one sample with multiple labers. Use sigmoid function
- USe gifflar code benchmarks.py and look for other data and code. code to turn pairs into multi-label
- Perhaps you could optimize a language model to produce nice embeddings to use in another model? '
- How do we know that the GLM actually produces high-quality embeddings?
- Use Kingdom prediction (from Daniel via Roman)
- Deeprank paper only ran their models for 20 models, was this to give the language model enhanced version the upper hand? what would happen after more epochs?
- How you handle embeddings for less frequent or unseen tokens might become a relevant consideration later
- Might need to downsample dominant organism groups to not skew data
- Concatenate the output of self.item_embedding(x) with the GLM embeddings
- Might I be making a 320 dimensional embedding where each dimension is a 320 dimensional vector?
- I should remember to clear outputs before commits for better edit history in the notebook
- 


# Ideas for Interesting applications (for framing in paper)
- Glycans as biomarker for disease x
  - Find disease in dataset that is hard to diagnose using traditional methods 
    - Parkinson's Disease: This is primarily a clinical diagnosis based on observing motor symptoms like tremor and slowness of movement. There is currently no definitive blood test or imaging marker that can definitively diagnose Parkinson's, especially in its early stages when symptoms are subtle and overlap with other movement disorders, leading to misdiagnosis. Glycan changes are being investigated in neurodegenerative diseases, and identifying specific glycan patterns in blood or other biofluids could potentially lead to objective biomarkers for earlier and more accurate diagnosis of Parkinson's disease and help distinguish it from similar conditions.
    - This case study provides a perfect opportunity to incorporate your SHAP analysis or other explainability methods, showing why the model made a certain prediction for that specific glycan and disease association. (make this one section of paper and perhaps indicate in title)
    - Pancreatic Cancer and Autoimmune Pancreatitis (AIP): Diagnosing AIP is challenging primarily because its symptoms and imaging findings can closely mimic those of pancreatic cancer. Distinguishing between these two conditions is crucial because their treatments are vastly different. 
  - Use SHAP analysis to find what features (sugars or bonds) influence the prediction of a specific disease the most, to get a general marker that could be easy to look for
- Glycans and caries
  - Might one develop a "toothpaste" that includes glycans that lower the risk of caries?
  - Using a slightly modified version of prep_infused _sweetnet as a general way to test infusion on different Gnn arcgitectures and embedding data sources? a model-agnostic infusion function
-  

# Improved model
- Try to use evolutionary algorithms for feature optimization (might be too slow)
  - Use grid search instead
  - raytune perhaps?
- Look into optimizing the feature matrix
- Explore different embedding fusion strategies (e.g., sum, learned layer, gating).
- Investigate impact of using embeddings from different GLM layers.
- Analyze sensitivity to embedding dimensionality or use dimensionality reduction.
- GGNN Uses gated recurrent units (GRUs) for node updates, allowing for longer-range information propagation.
- Implement Cross-Validation: K-fold cross-validation involves splitting your data into K folds. You train the model K times, each time using K-1 folds for training and the remaining fold for validation. This uses your data more efficiently and provides a more robust estimate of performance by averaging results over different train/validation splits.

# Analysis & Interpretation Ideas
- Visualize and analyze the learned SweetNet representations (e.g., using t-SNE, UMAP).
- Quantify and compare the performance of the standalone GLM, baseline SweetNet, and GLM-infused SweetNet ("emergent accuracy").
- Assess the computational cost trade-off of using GLM embeddings.
- Apply explainability methods (e.g., SHAP, GNN tools) to understand model predictions.
- Evaluate the transferability of the GLM-infused approach to other glycan prediction tasks/datasets.
- Benchmark the GLM-infused SweetNet against other state-of-the-art glycan representation learning methods (e.g., GIFFLAR).
- Compare GLM infusion impact across multiple prediction tasks (e.g., disease, tissue, different taxonomic levels) to assess generalizability.
- Use SHAP analysis to find what features (sugars or bonds) influence the prediction of a specific disease the most, to get a general marker that could be easy to look for 
- GNNExplainer 
- compare performances of Random and OneHot Embeddings, both fixed and learnable. And in case of learnable enbeddings, you can make tSNE or UMAP plots of the learned embeddings
  - For the tSNE/UMAP it's probably best to focus on the common monosaccharides or groups of monosaccharides with Same modifications, stuff like this, because you'll have ~2500 "token" in lib, i.e. classes in SweetNet nodes
- Multilabel NCC and AUROC


# Presentation (monday, can be adapted to main presenation later)
- present findings in tables for ease 
- Disease association data not complete, focus on other datasets (include but with caveat)
  - Kingdom, tissue,
- Sum up LRAP instead of averaging them
- Embeddings on base sweetnet might be way bigger (lib size)
  - I might have compared the bigger base Sweetnet to a reduced model.
  - No they seem to match after all
- tSNP plots for baseline trained embeddings, glm embeddings, and trained glm embeddings  for only the ones with icon
- Doubleheck dimensionality of SweetNet 
- Tables of results,
- Ideas for what is next
  - SHAP
  - test With other models (gifflar?)
  - increase model size
  - Hyperparameter Tuning
  - Parkinson biomarkers (not feaasible due to incomplete dataset?)
  - Run with differerent embeddings
  - Paper, statistics, research, and figures
- don't compare to old Sweetnet
- Present performance and speed (compare other method of infusion using dataloaders)
  - Compare infused model vs baseline
- Things to present
  - Intro to infued models
  - Pipelines
  - Diagram of how I run my models
  - different functions I've made. 
  - preliminary data
  - tSNP plots
  - 

# Ideas for paper
- GLMs take a long time to train and are resource intensive to run, by taking their embeddings we can improve efficiency while getting a model with emergent accuracy (better than both models that go in)
- Report on accuracy of GLM as well
- Hybrid approach
- Feature engineering
- Do I go into detail how glycowork transforms data into graphs and all that jazz?
- Should I compare my method to traditional methods to simulate glycans (Do they even exist?)
- Focus on case where SweetNet already outperforms gifflar ( and all other models )
- Diagram showing how the features from a GLM map into the architecture of the GNN
- In your discussion section, briefly touch upon potential applications of your method in areas like biomarker discovery. You can mention Disease X as a hypothetical example and discuss the potential benefits of using GLM embeddings for such tasks. This shows the broader relevance of your work without requiring you to solve a whole new problem.
- Negative result: Frame your discussion around the potential limitations of directly using language model embeddings for GNN initialization. You could suggest that the effectiveness of this approach might depend on the similarity between the language model's pre-training domain and the downstream task
- Positive result: Propose that this approach could be applicable to other domains where molecules or structured data can be represented as graphs and where language models can learn meaningful embeddings of the constituent parts 
- Regardless of the outcome, the key to making your findings generalizable is to think critically about why you got the results you did and to relate those reasons to broader principles of machine learning, graph representation learning, and transfer learning from language models.
- This kind of high-level framing can elevate your thesis beyond a specific application and contribute to the understanding of when and why these methods might be effective (or not) in a wider range of problems. It's definitely a worthwhile goal if you have the bandwidth to explore the "why" behind your findings.
-  I could use SHAP values for evaluation in my paper (as well as some nice plots)
-  Generate graphs for split (and discuss why I used stratified shuffle): https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
- Use cross-evaluation (edit multilabel_split)
- Multi-label nature of your task: Ensure your evaluation metrics and loss function are appropriate for multi-label classification.
- Graph data: Emphasize the use of GNNs to handle the non-linear structure of glycans.
- GLM embeddings: Clearly explain how you obtained and integrated these.
- Frame the results around the multi-task comparison: Show how GLM infusion performs across different types of glycan prediction problems.
- Discuss the implications of results across tasks – are some tasks more (or less) amenable to improvement via GLM embeddings, and why?
- I'm using the same exact prep funciton for  a fair comparison to truly isolate the impact of the GLM embeddings, ensuring that the rest of the SweetNet model (the GraphConv layers, linear layers, batch norms) starts from the same point in both the baseline and the GLM-infused versions
- Try to make infusion a thing, a general name for using pre-trained embeddings from a nother type of model. Include definition.
-  explainable AI (XAI)
-  Lay bare the forking paths that lead to my conclusions, map the model space that show different conclusions 
-  Glycans are the most information dense and complex biological sequences 
-  The sugar language of cells 
-  Glycans as the dark matter of biology 
-  Start with why and end with what when presenting in text or in person 
-  What I am doing is transfer learning
-  - Rather than using the more standard way to handle OOV by setting the glycowords not in the dictionary to zero, I will initialize all to random variables and only replace the ones I have embeddings for, to keep the comparison as clean as possible (I don't want the zero vectors to negatively influsence my results)
   -  Which was completely uneccecary, since the model doesn't really work that way, OOV is handled in data preparation
-  catastrophic forgetting
- Robust Finding: Your initial observation that the baseline model performs better seems to hold even after this round of hyperparameter adjustment for both models. This makes your finding more robust because you've attempted to optimize both configurations to a similar learning rate
- Based on preliminary learning rate sweeps, a rate of 0.005 was found to be effective for both the baseline and the infused-trainable models

# things to improve in glycowork
- documentation for using pretrained models
- links to examples in documentation broken
- the AUROC thing in model_training.py is broken, remove or fix
- I seem to get single-class batches in my training and validation data (bug uncovered while tinkering with auroc code)
- https://github.com/BojarLab/glycowork/blob/master/05_examples.ipynb Deep learning code snippets isn't correct, the model isn't for category prediction, the data needs to be transformed for multi-label prediction
- Fix LRAP at the end (takes overall best rather than best average, making it artificially high)
