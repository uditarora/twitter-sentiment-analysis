# twitter-sentiment-analysis
A simple console application that fetches and analyzes the sentiment of tweets

## Usage
These instructions assume that Python3 is set as the default python in your environment.

To fetch tweets from the Twitter API, you'll need to update the app key, secret and token in `twitterclient.py`

* To use the pretrained model with a default query of "#RoyalWedding":
```bash
$ python main.py
```

* To use the pretrained model with your own custom tweets, place the tweets in a text file (see tweets.txt for an example) and run:
```bash
$ python main.py --tweets <path_to_text_file>
```

* To force the model to retrain:
```bash
$ python main.py --train
```

* To see the evaluation metrics while training:
```bash
$ python main.py --train --eval
```

* To use the twitter API to fetch tweets for a given query:
```bash
$ python main.py --query "query"
```
