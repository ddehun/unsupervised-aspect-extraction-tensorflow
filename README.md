# Unsupervised-aspect-extraction-tensorflow
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
python preprocess.py
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
|  5   |                 |
|  10  |                 |
|  15  |                 |
|  20  |                 |
|  25  |                 |
|  30  |                 |
|  35  |                 |
|  40  |                 |
|  45  |                 |

2) Representative Words (sorted)

| Aspect ID |   Words   |
| --------- | ---- |
| 1         |      |
| 2         |      |
| 3         |      |
| 4         |      |
| 5         |      |
| 6         |      |
| 7         |      |
| 8         |      |
| 9         |      |
| 10        |      |
| 11        |      |
| 12        |      |
| 13        |      |
| 14        |      |
| 15        |      |



## Remaining Implementation

- Aspect identification evaluation metric based on labeled dataset
- Hyperparameter tuning
- Aspect embedding matrix initialization with k-means algorithm like original paper



## References

- An Unsupervised Neural Attention Model for Aspect Extraction(ACL 2017), Ruidan et al. [(pdf)](http://aclweb.org/anthology/P/P17/P17-1036.pdf)

- Pytorch implementation by the author of original paper. [(Github)](<https://github.com/ruidan/Unsupervised-Aspect-Extraction>)
