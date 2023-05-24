import os
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

"""
Script for experimenting with the langchain library that wraps a bunch of different LLMs,
and has implementations of various prompt engineering strategies.
In this case, we focus on the question and answer use case outlined here:
    https://docs.langchain.com/docs/use-cases/qa-docs
    
1. Pull Data from wikis in .md format mined from the Octopus Hydrogen notion.
2. Split into chunks of ~500 tokens and embed these into a document vector store using the GPT-2 embeddings
3. Given a question, embed it using two strategies (A. basic, and B. hypothetical document) and search the vector store
4. Ask GPT-3.5 to answer the question in both cases, given the context of the top 4 most relevant documents.  
"""

os.environ['OPENAI_API_KEY'] = open('/Users/samyoung/.openai-api-key.txt').read()

fpath = '/Users/samyoung/code/openai-cookbook/apps/hydrogen-q-and-a/fine_tuning_wikis'
wiki_paths = list(Path(fpath).rglob('*.md'))
wikis = [open(wiki_path).read() for wiki_path in wiki_paths]

markdown_splitter = MarkdownTextSplitter(chunk_size=5000, chunk_overlap=0)
texts = markdown_splitter.split_text(wikis)


# Based on https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
embeddings = OpenAIEmbeddings(model='gpt2')
docsearch = Chroma.from_texts(texts, embeddings).as_retriever()


queries = [
    'What decisions were taken about concurrency around a single event?',
    'How does AWS respond to suspicious activity?',
    'What architectural decisions were taken about ramping the electrolyzer up and down?',
    'What is the proposed architecture for IoT data pipeline',
    'What is dead reckoning when it comes to hydrogen?',
    'What is the imbalance price?',
    'How to choose a partition key for kinesis data streams?',
    'How does the backend app handle concurrent schedule requests?',
    'How does the EFA trading calendar change on long clock change day?',
    'what ORM tools are available to speedup backend development?'
]
query = queries[-1]


docs = docsearch.get_relevant_documents(query)
chain = load_qa_chain(OpenAI(temperature=0), chain_type='stuff')
chain.run(input_documents=docs, question=query)

# See: https://python.langchain.com/en/latest/modules/chains/index_examples/hyde.html
multi_llm = OpenAI(n=4, best_of=4)
hypothetical_embeddings = HypotheticalDocumentEmbedder.from_llm(multi_llm, embeddings, "web_search")
hypothetical_docsearch = Chroma.from_texts(texts, hypothetical_embeddings)

docs2 = hypothetical_docsearch.similarity_search(query)
chain2 = load_qa_chain(OpenAI(temperature=0), chain_type='stuff')
chain2.run(input_documents=docs2, question=query)
