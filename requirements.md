Top level requirements (Plan for whole project)

2.1.1 Must have:
1.	+Getting SweetNet Up and running (done 2025-04-16)
2.	!Infusing SweetNet with GLM Embeddings and comparing it (2025-04-25) <-- we are here
3.	Through testing and statistics to quantify how the infusion influenced the model (2025-05-02)
4.	Write a thesis that conveys my findings and results (2024-05-29)
2.1.2 Should have:
5.	Applying SweetNet to specific problem 
6.	Test embeddings from other GLMS
    1. Roman has at least 19
    2. Could try to extract embeddings from other models as well, such as SweetTalk
7.	Hyperparameter Optimized Sweetnet (Evolutionary algorithm?)
8.	Fix glycowork bugs
9.	Improve glycowork documentation
10. SHAP analysis (use GNNShap?)
2.1.3 Could have:
10.	Tinkering with Sweetnet Structure (birthing new model)
11. Pre-trained GLM-Infused SweetNet
2.1.2 Won’t have at this time (for future projects):
1.	



Current Iteration requirements [1]:

3.6.1 Must haves (60% of effort max)
0.	+Jog memory by rereading DeepRank paper
1.	+Explore Embedding data
2.	+Copy SweetNet Code to new Jupyter notebook for experimentation
3.	-Filter and Transform embedding data to a format usable in SweetNet
99. +I will need to modify my base sweetnet training code as well as implement data loading to change from categorical to multilabel
100. !While waiting for new embeddings
     1.   +Get dataloading and embedding pipeline working with glycowork data
     2.   +especially tissue data
     3.   !Modify embedding function with setting to not use fixed embeddings <-- we are here
     4.   look into lectin glycan interaction
     5.   refactor code
          1.   move finished functions into utils.py
          2.   make classes as needed
          3.   New clean notebook for further work
          4.   Do additional research
          5.   Rest
          6.   
4.	!Modify SweetNet to accept embedding data <-- stalled while awaiting new embeddings
    4.1	!Run training and compare
    4.2	If results don’t show improvement 
    4.3 look for errors or try other embeddings
    4.4 Go back to 4.1
3.6.2 Should haves
5.	Add new SweetNet functionality to glycowork 
6.	Test other GLM embeddings (if the first one fails)
7.  !Work on paper <-- we are here as well

3.6.3 Could haves (20% of effort)
7.	
3.6.4 Won’t have at this time (for future iterations)
8.	


3.7 What is the done definition of each requirement? 
0.	I’ve read the text again
1.	I understand the structure and contents of the data
2.	I can run the local experiment and get the same results as the imported version
3.	I get a fitting datastructure that I can pipe into SweetNet
4.	I can run training using the modified SweetNet and get an accuracy score
5.	I port my experimental code into the models.py file of glycoworks without breaking anything
6.	See if any of the other embeddings give different results
3.7.1 How could each requirement be tested?
0.	Do I Understand what I need to be doing?
1.	What is the data I need? How might I add it?
2.	Running the kingdom test with copied sweetnet, see that it behaves similarly
3.	I get data that works in the modified sweetnet
4.	Training works (I get convergence and accuracy scores)
5.	Training works when running from local glycowork, 
a.	test with other settings to make sure nothing broke

