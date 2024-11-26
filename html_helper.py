import os
from llmlingua import PromptCompressor
from termcolor import cprint
from transformers import AutoTokenizer
from trafilatura import extract
from bs4 import BeautifulSoup, NavigableString
import nltk
import requests
import re
import tiktoken


def download_html(url, file_path):
    """
    Downloads HTML content from a URL and saves it to a file if it doesn't already exist.
    If the file exists, it reads the content from the file.

    Args:
        url (str): The URL to download the HTML content from.
        file_path (str): The path to the file where the HTML content will be saved.

    Returns:
        str: The HTML content.
    """
    if not os.path.exists(file_path):
        response = requests.get(url)
        html_content = response.text
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        #print(f"HTML content downloaded and saved to {file_path}")
    else:
        #print(f"{file_path} already exists. Skipping download.")
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

    return html_content

def get_page_body_text(raw_page):

    # If raw_page is None or empty, return False
    if not raw_page:
        print("get_page_body_text - raw_page is None or empty")
        return False

    #add line breaks to the end of certain tags so that there are proper linebreaks in the clean output text
    soup = BeautifulSoup(raw_page, 'html.parser')

    tags = ['ol', 'ul', 'li', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    for tag in tags:
        for match in soup.find_all(tag):
            match.append('\n\n')

    # Find all text nodes that are direct children of the body
    for elem in soup.body:
        if isinstance(elem, NavigableString) and elem.strip():
            # Wrap the text node in a p tag
            new_tag = soup.new_tag("p")
            new_tag.string = elem
            elem.replace_with(new_tag)
        # Use the get_text method to extract all the text, stripping away the HTML
        text = extract(soup.prettify(), favor_recall=True)

    clean_text = re.sub('\n-\s*\n', '\n- ', text)
    clean_text = re.sub('\s*\n+\s*', '\n\n', clean_text)
    clean_text = clean_text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    # Return the cleaned text
    return clean_text

def initialize_compressor():
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    encoding = tiktoken.get_encoding("cl100k_base")
    return llm_lingua, tokenizer, encoding

def compress_prompt(text, llm_lingua, tokenizer, encoding, compression_ratio=0.5, debug=False):
    """
    Compresses a given text prompt by splitting it into smaller parts and compressing each part.
    
    Args:
        text (str): The text to compress.
        llm_lingua (PromptCompressor): The initialized PromptCompressor object.
        tokenizer (AutoTokenizer): The initialized tokenizer.
        encoding (Encoding): The initialized encoding.
        compression_ratio (float): The ratio to compress the text by.
        debug (bool): If True, prints debug information.
    
    Returns:
        str: The compressed text.
    """
    if debug:
        print(f"Compressing prompt with {len(text)} characters")

    # Split the text into sentences
    sentences = nltk.sent_tokenize(str(text))
    
    compressed_text = []
    buffer = []

    for sentence in sentences:
        buffer_tokens = encoding.encode(" ".join(buffer))
        sentence_tokens = encoding.encode(sentence)

        # If the sentence exceeds the token limit, split it
        if len(sentence_tokens) > 400:
            if debug:
                print(f"Sentence exceeds token limit, splitting...")
            parts = split_sentence(sentence, encoding, 400)
            for part in parts:
                part_tokens = encoding.encode(part)
                if len(buffer_tokens) + len(part_tokens) <= 400:
                    buffer.append(part)
                    buffer_tokens = encoding.encode(" ".join(buffer))
                else:
                    if debug:
                        print(f"Buffer has {len(buffer_tokens)} tokens, compressing...")
                    compressed = llm_lingua.compress_prompt(" ".join(buffer), rate=compression_ratio, force_tokens=['?', '.', '!'])
                    compressed_text.append(compressed['compressed_prompt'])
                    buffer = [part]
                    buffer_tokens = encoding.encode(" ".join(buffer))
        else:
            # If adding the sentence exceeds the token limit, compress the buffer
            if len(buffer_tokens) + len(sentence_tokens) <= 400:
                if debug:
                    print(f"Adding sentence with {len(sentence_tokens)} tokens, total = {len(buffer_tokens) + len(sentence_tokens)} tokens")
                buffer.append(sentence)
            else:
                if debug:
                    print(f"Buffer has {len(buffer_tokens)} tokens, compressing...")
                compressed = llm_lingua.compress_prompt(" ".join(buffer), rate=compression_ratio, force_tokens=['?', '.', '!'])
                compressed_text.append(compressed['compressed_prompt'])
                buffer = [sentence]

    # Compress any remaining buffer
    if buffer:
        if debug:
            print(f"Compressing final buffer with {len(encoding.encode(' '.join(buffer)))} tokens")
        compressed = llm_lingua.compress_prompt(" ".join(buffer), rate=compression_ratio, force_tokens=['?', '.', '!'])
        compressed_text.append(compressed['compressed_prompt'])

    result = " ".join(compressed_text)
    if debug:
        print(result)
    return result.strip()