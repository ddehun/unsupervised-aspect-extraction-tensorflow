# unsupervised-aspect-extraction-tensorflow
- Tensorflow reimplementation of ACL 2017 paper "An Unsupervised Neural Attention Model for Aspect Extraction"[(pdf)](http://aclweb.org/anthology/P/P17/P17-1036.pdf)  for practice.



## Requirements

- python=3.6

- numpy==1.16.2

- nltk==3.3

- tensorflow_gpu==1.8.0

- tqdm

- matplotlib

  

## Package Structure

```
|
├── main.py  # Training & evaluation main script. Containing Hyperparameters.
├── model.py  # Model
├── dataset.py  # Batching
├── preprocess.py  # Preprocess the raw dataset & Serialize into binary file.
├── utils.py  # Utility functions
├── data/
├── model/
```



## Usage

### Data

- Download the unpreprocessed review dataset from author's [(github)](<https://github.com/ruidan/Unsupervised-Aspect-Extraction>).
- Download the pretrained Glove word embedding [(Glove)](<https://nlp.stanford.edu/projects/glove/>)

```
mkdir data
# Decompress the review dataset into here.
# Put in the 'glove.6B.200d.txt' file here.
python preprocess.py --dataset=[restaurant, beer]
```

### Train

```
python main.py --mode=train
```

### Test

```
python main.py --mode=test
```



## Results 

- Training and evaluation is based on restaurant review corpus (Citysearch corpus) only.

1) Coherence Score (along with K)

|  K   | Coherence Score |
| :--: | :-------------: |
|  5   |     -7.3815     |

2) Representative Words (sorted)

| Aspect ID |   Words   |
| --------- | ---- |
| 1         | lombardis dissapointing coffe flautas geido |
| 2         | recomment bannana loungy arugala bottomless |
| 3         | cheescake veniero saganaki trully ideya |
| 4         | wondee disapointment bernaise housemade curtious |
| 5         | 30pm 00pm deliscious omlettes goal |
| 6         | pleasent carnivorous brushetta bouterin servce |
| 7         | 30pm atmostphere shortribs suace cannolis |
| 8         | margharitas prixe amzing gnudi chikalicious |
| 9         | <PAD> imho overated poetry genre |
| 10        | parmesean fusia accomadating molyvos tabouleh |
| 11        | kababs octupus shortribs foccacia higly |
| 12        | kittichai markjoseph aweful oversalted soccer |
| 13        | barmarche ofcourse sauted waitperson negimaki |
| 14        | waittress peices phenominal ramblas sandwhich |
| 15        | moqueca pampano perbacco absolutly dissappointing |



## Remaining Implementation

- Aspect identification evaluation metric based on labeled dataset
- Hyperparameter tuning
- Aspect embedding matrix initialization with k-means algorithm like original paper
- NaN issue at 35k step



## References

- An Unsupervised Neural Attention Model for Aspect Extraction(ACL 2017), Ruidan et al. [(pdf)](http://aclweb.org/anthology/P/P17/P17-1036.pdf)

- Pytorch implementation by the author of original paper. [(Github)](<https://github.com/ruidan/Unsupervised-Aspect-Extraction>)
