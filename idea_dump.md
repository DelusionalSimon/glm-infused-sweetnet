
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


# Ideas for Interesting applications (for framing in paper)
- Glycans as biomarker for disease x

# Improved model
- Try to use evolutionary algorithms for feature optimization (might be too slow)
  - Use grid search instead
  - raytune perhaps?
- Look into optimizing the feature matrix
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
- 

# things to improve in glycowork
- documentation for using pretrained models
- links to examples in documentation broken
- the AUROC thing in model_training.py is broken, remove or fix
- I seem to get single-class batches in my training and validation data (bug uncovered while tinkering with auroc code)
- https://github.com/BojarLab/glycowork/blob/master/05_examples.ipynb Deep learning code snippets isn't correct, the model isn't for category prediction, the data needs to be transformed for multi-label prediction
- 
