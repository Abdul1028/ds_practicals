import gensim
from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint

# Sample documents
documents = [
    "Machine learning algorithms can analyze large datasets to find patterns.",
    "Natural language processing enables computers to understand and generate human language.",
    "Data scientists use statistical methods to extract insights from data.",
    "Artificial intelligence is revolutionizing various industries such as healthcare and finance.",
    "Deep learning models are capable of learning complex patterns from raw data."
]


# Tokenize the documents
tokenized_documents = [document.lower().split() for document in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(tokenized_documents)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Build LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=10)

# Print the topics
pprint(lda_model.print_topics())

# Get topic distribution for a document
doc_lda = lda_model[corpus[0]]
print("\nTopic distribution for the first document:")
pprint(doc_lda)
