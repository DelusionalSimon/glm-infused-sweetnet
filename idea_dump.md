
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


# Ideas for Interesting applications (for framing in paper)
- 

# Improved model
- Try to use evolutionary algorithms for feature optimization
- Look into optimizing the feature matrix
- 

# Ideas for paper
- GLMs take a long time to train and are resource intensive to run, by taking their embeddings we can improve efficiency while getting a model with emergent accuracy (better than both models that go in)
- Report on accuracy of GLM as well
- Hybrid approach
- Feature engineering
- Do I go into detail how glycowork transforms data into graphs and all that jazz?
- Should I compare my method to traditional methods to simulate glycans (Do they even exist?)
- Focus on case where SweetNet already outperforms giifflar ( and all other models )

# things to improve in glycowork
- documentation for using pretrained models
- links to examples in documentation broken
- the AUROC thing in model_training.py is broken, remove or fix
- 
