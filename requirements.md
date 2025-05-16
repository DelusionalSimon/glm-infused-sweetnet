Using MoSCoW Prioritization

Top level requirements (Plan for whole project)

*Must have:*
1.	+Getting SweetNet Up and running (2025-04-11 2025-04-16)
2.	+Infusing SweetNet with GLM Embeddings and comparing it (2025-04-25 2025-05-12)
3.	!!Through testing and statistics to quantify how the infusion influenced the model (2025-05-23)
4.	!Write a thesis that conveys my findings and results (2024-05-29) (see paper writing catech)
*Should have:*
5.	More generalizable results 
    5.2	general "infuse" function to test different GNN architectures
        5.2.1	Test with Lectinoracle
6.	Applying SweetNet to specific problem 
7.	Hyperparameter Optimized Sweetnet (Evolutionary algorithm?)
8.	Improve Glycowork
    8.2	Fix glycowork bugs
    8.3	Improve glycowork documentation
*Could have:*
9.	Tinkering with Sweetnet Structure (birthing new model)
10.	13. One-hot encoding tests
	



Current Iteration [2]:

I’m going to thoroughly test my infusion method using several different tasks and datasets, running through statistical analyzes on the data that I generate to quantify how infusion affects the performance of the model. I’m also going to try to figure out why I get the results I get by looking at the embeddings using t-SNE and perhaps by using SHAP or some other explainability framework adapted to GNNs. Througout this I will generate tables, statistics, and graphs that I will use in my paper that I will be writing in paralell to my statistical analysis. Focusing on a MVP version that I can iterate using nicer graphs and better language. 

---- Requirements -----

**Must haves (60% of effort max)**
0.	Core Data collection pipeline
    0.2	Change test split pipeline, splitting out test set before experiment and saving it
        0.2.1	Add flag to split function to split just once and use twise, one time outside of loop and then within the loop
        0.2.2	Perhaps doing the more standard way and just splitting before each experiment and just rerunning training. Compare variability
        0.2.3	Save test set for each experiment, and other split used for further split down the line
    0.3	Save models 
        0.3.1	Save the model.state_dict() of every training run
        0.3.2	Run sufficient experiments with modified pipeline (10 x disease, kingdom, and tissue)
1.	Core statistical analysis to quantify how infusion influences the model
    1.2	t-SNE Analysis
        1.2.1	notebook that loads embeddings
    1.3	Statistical analysis of the three experiments I have run thus far
        1.3.1	Delve into what kind of methods might be useful
        1.3.2	Delve into pandas
        1.3.3	make analysis notebook that takes raw data from my automated system
            1.3.3.1	gives averages +-sd  for tables
            1.3.3.2	Quantifies the difference between treatments
    1.4	Write relevant part of paper as each analysis is done
2.	Core Tables and diagrams (run in parallel withs statistical analysis)
    2.2	Delve into plotting libraries in python
    2.3	Basic t-SNE diagrams
    2.4	Main results table comparing infusion vs random for different tasks
        2.4.1	Regenerate table from presentation using Python
        2.4.2	Populate with new data as it comes in
    2.5	Add to paper and write explanations when each is done
3.	Find best model and test with test set
    3.2	Use new data with saved models
    3.3	Add a system to find the best models in the data
    3.4	At the very end, once I know I won’t collect more data, test the best model I’ve found over all using the test set that I have also saved for each prediction task I have explored

**Should haves**
4.	Advanced Data collection [run in parallel with other requirements]
    4.2	Set up desktop-based data collection system that can run all the time
        4.2.1	Hyperloop to run several experiments one after another
    4.3	Collect data from several different tasks
        4.3.1	Different df_species hierarchies
    4.4	Get performance data of GlyLMs from roman
    4.5	Try embeddings from other GLyLMs (trivial with the pipeline I have)
5.	More advanced statistical analysis
    5.2	Compare the t-SNE plots of several random embeddings to see how much they differ
    5.3	SHAP analysis (use GNNShap?)
6.	Advanced diagrams and tables
    6.2	Infusion diagram
    6.3	Delve into other ways to represent my findings?
    6.4	Make sure graphs are well designed
**Could haves (20% of effort)**
7.	Optional Diagrams 
    7.2	Sweetnet diagram (could be included in infusion diagram)
    7.3	Graphical abstract?
    7.4	Graphs for split
    7.5	Testing system diagram
    7.6	Beautify Graphs
8.	Optional Data collection
    8.2	Once I have collected all the data I need, run simple hyperparameter exploration
        8.2.1	To give my desktop something to do, setting up each experiment takes no time
    8.3	Collect data from other prediction types
        8.3.1	Glycolisation
        8.3.2	Basic category kingdom
9.	Optional statistical analysis
    9.2	Comparing several different statistical methods
    9.3	Metanalysis of several prediction tasks
    9.4	Compare performance to the state of the art
        9.4.1	Models of similar size




--- Done Definitions -----

**3.6.1 Must haves**

* **0. Core Data collection pipeline:** The pipeline is successfully modified to save metrics, test splits, and models, and the necessary experiments (10x Disease, Kingdom, Tissue) are completed using this modified pipeline.
* **1. Core statistical analysis to quantify how infusion influences the model:** The main statistical analysis (comparing configurations on Disease, Kingdom, Tissue) and basic t-SNE analysis are completed, providing quantitative measures of performance differences and initial visual interpretations.
* **2. Find best model and test with test set:** The best overall model is identified based on the statistical analysis, and its performance on the saved test set is successfully calculated.
* **(Implicit requirement based on 0.3 and 1.5) Write relevant part of paper / Core Tables and diagrams:** The main results tables and basic diagrams (like the performance table and t-SNE plots) are generated using Python, and the corresponding sections of the paper are drafted.

**3.6.2 Should haves**

* **4. Advanced Data collection:** The infrastructure and processes for running more extensive data collection experiments are set up and functional.
* **5. More advanced statistical analysis:** Deeper statistical analysis or interpretability methods (like comparing random t-SNEs or SHAP) are successfully applied.
* **6. Advanced diagrams and tables:** More complex or polished diagrams and tables beyond the core outputs are generated.

**3.6.3 Could haves**

* **7. Optional Diagrams:** The specific optional diagrams are successfully created.
* **8. Optional statistical analysis:** The intended optional statistical analyses are successfully run and provide results.
* **9. Optional Data collection:** The intended optional data collection (HPO or other prediction types) is successfully completed.






----- Tests -----
**3.6.1 Must haves**

* **0. Core Data collection pipeline:**
    * **Test 1:** Can the main pipeline script be successfully executed from start to finish for a single experiment configuration (e.g., Baseline Trainable on Kingdom data) without crashing or errors?
    * **Test 2:** After running the pipeline for a single experiment run, are the following output files successfully generated with unique identifiers linked to the configuration and run number: metric summary file, test set split file, and best model state dictionary file?
    * **Test 3:** After running "sufficient experiments" (e.g., 10 runs for one dataset/config), are you able to load the collected metric data (e.g., from CSV/PKL) and verify that there is data recorded for each run?
* **1. Core statistical analysis to quantify how infusion influences the model:**
    * **Test 1:** Does the analysis notebook successfully load the results data from your completed experiment runs without errors?
    * **Test 2:** Does the analysis notebook successfully calculate the mean and standard deviation for key metrics (e.g., Validation LRAP, Loss) for each configuration across the runs?
    * **Test 3:** Does the analysis notebook successfully perform the selected statistical tests (e.g., t-tests) comparing the performance metrics between configurations and output the results (e.g., p-values)?
    * **Test 4:** Are you able to generate basic t-SNE plots comparing the embeddings from representative models without errors?
* **2. Find best model and test with test set:**
    * **Test 1:** Can you successfully load a saved model state dictionary and the corresponding saved test set data (or indices)?
    * **Test 2:** Can the `test_model` function be successfully run using a loaded best model and its corresponding test set data?
    * **Test 3:** Does the `test_model` function output the expected set of metrics (Loss, LRAP, NDCG, etc.) for the test set?

*Note: Requirements like "Write relevant part of paper" and "Core Tables and diagrams" are outputs of the analysis and data collection steps. Their testing might be more about review:*

* **Write relevant part of paper:**
    * **Test 1:** Is the relevant section of the paper drafted, incorporating the findings from the analysis and outputs? (Verification through self-review or review by Roman).
* **Core Tables and diagrams:**
    * **Test 1:** Are the main results tables and basic diagrams (like the performance table and a basic t-SNE plot) successfully generated using Python based on the analysis results?
    * **Test 2:** Are these tables and diagrams clear and understandable representations of the core findings? (Verification through self-review or review by others).

**3.6.2 Should haves**

* **4. Advanced Data collection:**
    * **Test 1:** Is the desktop-based data collection system successfully set up and can it run the pipeline?
    * **Test 2:** If implemented, does the hyperloop correctly run multiple experiments sequentially?
    * **Test 3:** After running experiments using the advanced pipeline, are data files, split files, and model state dictionaries successfully saved with the intended structure?
    * **Test 4:** If rerunning first experiments for reproducibility, do the results (metrics, maybe saved models) match closely with the initial runs (within expected variability)?
    * **Test 5:** After running experiments on new tasks or using other GlyLMs, are the results collected and structured correctly for analysis?
    * **Test 6:** Can performance data from Roman be successfully loaded and accessed?
* **5. More advanced statistical analysis:**
    * **Test 1:** If implemented, does the comparison of t-SNE plots across several random embeddings show clear patterns of variability or similarity?
    * **Test 2:** If implemented, can SHAP analysis be successfully run on a trained model for a sample prediction and output feature contributions?
* **6. Advanced diagrams and tables:**
    * **Test 1:** Are the intended advanced diagrams (like the infusion diagram) successfully created?
    * **Test 2:** Are the key graphs and tables reviewed and refined for clarity and visual appeal? (Verification through self-review or feedback).

**3.6.3 Could haves**

* **7. Optional Diagrams:**
    * **Test 1:** Are the specific optional diagrams (SweetNet, split, testing system, graphical abstract) successfully created?
    * **Test 2:** Are the graphs aesthetically polished? (Verification through self-review or feedback).
* **8. Optional statistical analysis:**
    * **Test 1:** Are the intended optional statistical analyses successfully run (e.g., comparing different stat methods, meta-analysis, state-of-the-art comparison)?
    * **Test 2:** Do these analyses yield results that add value to the project or paper? (Verification through self-review or discussion).
* **9. Optional Data collection:**
    * **Test 1:** If implementing hyperparameter exploration, can the HPO process be initiated and run successfully?
    * **Test 2:** If collecting data from other prediction types, are experiments successfully run and data collected for these tasks?
