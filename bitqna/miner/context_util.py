import random
import requests
from bs4 import BeautifulSoup

import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

from typing import List, Sequence

chroma_client = chromadb.Client()

# TODO these params should be configurable
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
)

# TODO URLs can point to anything
#       - initially we'd focus on requests with text/html
#       - then grow to:
#           - wildcard url like example.com/path/*
#           - dropbox file(s) (or the like)
#           - dropbox folder(s) (or the like)
#           - gz/zip file(s) (or the like)
#           - API end points for updated news (or the like)
#           - audio streams
#           - images
#           - videos
#           - etc.
#
# TODO for now check the url content type and handle accordingly
# TODO Citations are impacted by the content type
#       - citation for text/html would be a link to the highlighted content from the url
#       - citation for video would be a link to the most relevant timestamp(s) of the video
#       - citation for other types will vary
#       - default would be to provide the url back 
#           - as an example, for images, initially, the image would be the citation
#               - e.g., 100 image urls are provided, only 10 are relevant, provide those 10 urls
#           - but perhaps we reach a point where the citation is a selection of bounding boxes within
#               - e.g., where's waldo?  he's by the ice cream truck - note that part of the image for citation


def get_relevant_context_and_citations_from_urls(urls: List[str], prompt: str) -> List:
    if not urls:
        # if urls is empty, we don't have anything to do wrt context
        return ["",[]]

    # TODO n_results should be configurable
    collection = __index_data_from_urls(urls)
    results = collection.query(query_texts=[prompt],n_results=4)
    # TODO citations - should be a link to the highlighted text from the url
    # TODO example: http://example.com/page#text-to-highlight
    # TODO test these links in chrom/firefox/edge/etc.
    # return the relevant chunk data and the citation urls (TODO for citations)
    return [" ".join(results['documents'][0]), []]

def __index_data_from_urls(urls: List[str]) -> Sequence:
    collection = chroma_client.create_collection(name=__generate_collection_name())
    for url in urls:
        page = requests.get(url)
        soup = __extract_text_from_html(page)
        chunks = text_splitter.create_documents([soup])
        docs = [c.page_content for c in chunks]
        # TODO should we get more useful info to help with the citation links?
        collection.add(documents=docs, 
                       ids=["id"+str(i) for i in range(len(docs))],
                       metadatas=[{"source": url} for i in range(len(docs))])

    return collection

def __extract_text_from_html(page):
    soup = BeautifulSoup(page.text, 'lxml')

    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def __generate_collection_name() -> str:
    h = random.getrandbits(128)
    return f'bitqna.collection.{h}'
