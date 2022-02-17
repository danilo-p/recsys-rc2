import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from strsimpy.sorensen_dice import SorensenDice
from tqdm import tqdm
import sys

tqdm.pandas()

ratings_filename = sys.argv[1]
content_filename = sys.argv[2]
targets_filename = sys.argv[3]

# let r be number of ratings
ratings_df = pd.read_json(ratings_filename, lines=True)
# let c be number of items
content_df = pd.read_json(content_filename, lines=True)
# let t be number of targets
targets_df = pd.read_csv(targets_filename)

# read input complexity: O(r + c + t)

nltk.download('punkt') # O(1)

porter = PorterStemmer()

# let s be sentence size
# complexity: O(s)
def stem_sentence(sentence):
    words = word_tokenize(sentence)
    stemmed_words = [porter.stem(word) for word in words] # assuming constant complexity for stemming
    return ' '.join(stemmed_words)

# complexity: O(1)
def stem_column(column):
    # complexity: O(s)
    def stem_row(row):
        return stem_sentence(row[column])
    return stem_row

content_df['Title'] = content_df.apply(stem_column('Title'), axis=1)
content_df['Plot'] = content_df.apply(stem_column('Plot'), axis=1)
content_df['Genre'] = content_df.apply(stem_column('Genre'), axis=1)

# let S be the greatest sentence size
# stemming content complexity: O(c*S)

# let A be the number of cols (attributes) below
cols = ['Title', 'Genre', 'Plot', 'Writer', 'Director', 'Actors', 'Language', 'Country']

# merge complexity: O(r*c)
ratings_enriched_df = ratings_df.merge(content_df[['ItemId'] + cols], on='ItemId')

agg_dict = {}
for col in cols:
    agg_dict[col] = ', '.join

# let u be the number of users
# assuming linear complexity for groupby
# complexity of building users representation: O(r)
users_df = ratings_enriched_df.groupby('UserId').agg(agg_dict)
users_df.index.name = 'UserId'
users_df.reset_index(inplace=True)
users_df

def build_columns_dict(prefix, cols):
    cols_dict = {}
    for col in cols:
        cols_dict[col] = prefix + col
    return cols_dict

# complexity of enriching targets: O(t*u + t*c)
targets_enriched_df = targets_df\
  .merge(users_df.rename(columns=build_columns_dict('User', cols)), on='UserId')\
  .merge(content_df[['ItemId'] + cols]\
  .rename(columns=build_columns_dict('Item', cols)), on='ItemId')

# let n be the n-gram size
# let a be the size of sentence a
# let b be the size of sentence b
# complexity: O(a/n + b/n)
def ngram_dice(n, a, b):
    return 1 - SorensenDice(n).distance(a, b)

# let N be the constant below
n = 5
# complexity: O(A*S/N)
def create_score(row):
    score = 0
    for col in cols:
        if len(row['User' + col]) > n and len(row['Item' + col]) > n:
            score += ngram_dice(n, row['User' + col], row['Item' + col])
    return score

# complexity to calculate scores: O(t*A*S/N)
targets_enriched_df['Score'] = targets_enriched_df.progress_apply(create_score, axis=1)

# complexity to sort: O(t*log t)
predictions_df = targets_enriched_df[['UserId', 'ItemId', 'Score']]\
    .sort_values(by=['Score'], ascending=False)[['UserId', 'ItemId']]

# complexity to print: O(t)
print(predictions_df.to_csv(index=False))

# final complexity
# O(r + c + t) + O(c*S) + O(r*c) + O(r) + O(t*u + t*c) + O(t*A*S/N) + O(t*log t) + O(t)
# O(r + c + t + c*S + r*c + r + t*u + t*c + t*A*S/N + t*log t + t)
# O(c*S + r*c + t*u + t*c + t*A*S/N + t*log t)
# O(c*(S + r) + t*(u + c + A*S/N + log t))
# note: constants are considered to understand the impact of the number of attributes and n-gram size