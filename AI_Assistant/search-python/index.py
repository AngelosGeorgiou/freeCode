ASTRA_DB_SECURE_BUNDLE_PATH = "/home/angelos/myGitRepos/freeCode/AI_Assistant/search-python/secure-connect-vector-database.zip"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:tJTPLtffJkFXbFxXaCiHEtRW:557d4dda00d4ae4a92ed60a9789edc10112b11bc59929cb775b1f045556abe5c"
ASTRA_DB_CLIENT_ID = "tJTPLtffJkFXbFxXaCiHEtRW"
ASTRA_DB_CLIENT_SECRET = "OGok1Q+Kn4eUDRWiJd3Fuo35qaOfUECO5YWUNshazbdja+PMOZ3,22Znqtb7IOzBvDYb8TjEB-X8T_.8E++UtEZK-c_U3tqHR8GfF.tdAW.zj11cqntrdAixE-o1HMxY"
ASTRA_DB_KEYSPACE="search"
OPENAI_API_KEY="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"

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

cloud_config = {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini_demo"
)

print("Loading data from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\nGenerating embeddings and storring in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i headlines.\n" % len(headlines))

vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

first_question = True

while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ")
    
    if query_text.lower() == 'quit':
        break

    print("QUESTION: \"%s\"\n" % query_text)
    answer = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("DOCUMENT BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print(" %0.4f \"%s ...\"" % (score, doc.page_content[:60]))