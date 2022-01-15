# Aspect-Opinion Mining


The task is to create aspect-opinion co-occurrence sentiment heat map.
The key idea is to extract noun and adjective/adverbs from the input document and then build two LDA models, one based
only noun and the other based on adj/adv. Then by using LDA inference methods, calculate co-occurrence of noun topic and
adj/adv topic for every document.
The package is standalone, scalable, and can be freely extended to your needs.


<br>


### Data
You can find the pre-processed datasets and the pre-trained models in
[[Download]]().
The zip file should be decompressed in the main folder and run:
```bash
$ python Main.py --prep_noun noun.xlsx --prep_adj adj.xlsx \
```

You can also download the original datasets of Casino domain in 
[[Download]](). 
For preprocessing, put the decompressed zip file in the main folder and run 
```bash
$ python Main.py --path dataset.xlsx
```
The preprocessed files and lda model for each domain will
be saved in folders results/~ and models/~ respectively.

<br>

### Using Pre-Trained Models
Models can be built and reused, to do that, run:
```bash
$ python Main.py --tune False --noun_model pxp_model_noun.pxp \
                 --adj_model pxp_model_adj.pxp
```
<br>

### Tune
Number of extracted topics can be automatically detected by default:
or explicitly indicated, run:
```bash
$ python Main.py --tune False --num_topics 20
``` 
Tune logic is to first break dataset using KFold to 5 smaller parts
then iteratively calculate coherence value for each, using mean and std in the 
process of choosing optimal number of topics, here are results for noun dataset and adjective dataset:

<br>

##### For NOUN dataset: 
![results4](picture/coherence-topics-noun.png)
optimal number: 36 

<br>

##### For ADJ/ADV dataset:
![resutls5](picture/coherence-topics-adj.png)
optimal number: 39
<br>

### Result
Here are some visualization on our dataset of scraped 
google reviews for all Canadian casinos.

![results0](visual/cloud_casino1_no_collocations.png)

and final Aspect-Opinion Sentiment Co-occurrence heat map report:
 
![results1](picture/result_occurrence.png)

Note that each gray cell represent an irrelevant pair, these irrelevancies
are calculated using glove-twitter-50 word embedding.
Sentiment of each cell is calculated using TextBlob sentiment tool, green
spectrum stands for positive, and red spectrum for negative sentiments
<br>

### Inference
To derive results from trained models run:
```bash
$ python Main.py --inference inference_set.xlsx
```
note that models need to be indicated first, otherwise 
a model will be build from default settings.

Here, Aspect-Opinion Sentiment Co-occurrence heat map report
for Woodbine resort, inferred from the all Canadian casinos model:

![results2](picture/inference_woodbine_occurrence1.png)
Woodbine dataset contains ~2400 reviews.
<br>

### Dependencies

* python        3.8.8
* nltk          3.5
* scikit-learn  0.24.1
* spacy         3.0.5
* gensim        3.8.3
* textblob      0.15.3

Also to make use of Spacy language model, run
```bash
$ conda install -c conda-forge spacy

$ pip install spacy-transformers
$ pip install spacy-lookups-data

$ python -m spacy download en_core_web_trf
```
See also requirements.txt
You can install requirements, using the following command.

```bash
$ pip install -r requirements.txt
```

<br>

### Costs

* Memory-count < 2GB
* Working set < 1GB
* CPU Avg. cycle = 49.7
* Pipline actual duration is ~11 hours on a dataset of size ~6800 

<br>

##### Bottle necks: 
* Preprocess 77% (angelicized method used 60% of total resources), 
* Topic Modeling 20% (cross validation method used 18% of total resources)

### References

Developed by [Press'nXPress](https://pressnxpress.com/)
