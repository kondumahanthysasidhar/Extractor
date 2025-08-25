import streamlit as st
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration

import fitz
import io
import os
from crawl4ai import AsyncWebCrawler, WebCrawler
import html2text
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import pymupdf
import subprocess
import torch



def main():

    st.markdown(
        """
        <style>
        /* Set background color to white */
        body {
            background-color: white;
            font-family: 'Arial', sans-serif;
            color: #001F54; /* Navy Blue Text Color */
        }
         /* Styling for the Tab/Selectbox section */
    .css-1a93dvs {  /* Tab/Selectbox container */
        background-color: #001F54;  /* Navy Blue background */
        color: white;  /* Text color for the tab */
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }

    /* Style active tabs to highlight them */
    .css-1a93dvs .stSelectbox>div>div>div {
        background-color: #003366;  /* Darker Navy for active tabs */
        color: white;  /* Active tab text color */
    }

    .css-1a93dvs .stSelectbox>div>div>div:hover {
        background-color: #003366;  /* Hover effect for active tabs */
    }


        /* Header styling */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #001F54; /* Navy Blue */
            font-weight: bold;
        }

        /* Customize the sidebar */
        .css-1d391kg {
            background-color: #001F54; /* Navy Blue */
            color: white;
        }

        /* Style buttons */
        .stButton>button {
            background-color: #001F54; /* Navy Blue */
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #003366; /* Darker Navy Blue on hover */
            color: #FFF;
        }

        /* Input boxes styling */
        .stTextInput > div > input {
            border: 1px solid #001F54; /* Navy border */
            background-color: #f9f9f9; /* Light gray background */
            color: #001F54; /* Navy text */
        }

        /* Custom title styling */
        .title {
         display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh; /* Adjust height to center vertically */
        text-align: center;
        font-size: 2.5rem;
        color: #001F54;
        font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Add CSS to adjust st.image for a full-width banner
    st.markdown(
        """
        <style>
        .full-width-banner {
            display: block;
            width: 100%;
            height: auto;
            margin-bottom: 20px; /* Add spacing below the banner */
        }
         .tile-button {
            width: 100%;
            height: 150px;
            background-color: #001F54; /* Navy Blue */
            color: white;
            border-radius: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            cursor: pointer;
            padding: 20px;
            margin-bottom: 20px;
        }

        .tile-button:hover {
            background-color: #003366; /* Darker Navy Blue */
            transform: translateY(-10px);
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3);
        }

        .tile-button:active {
            background-color: #004080; /* Even darker navy for active state */
            transform: translateY(0);
        }

        .tile-button h3 {
            font-size: 1.5rem;
            margin: 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    # st.set_page_config(layout="wide")

    # Display the banner image
    st.image("ADP.jpg", use_container_width=True, output_format="PNG", caption="")

    # st.set_page_config(page_title="Extraction App", layout="wide")

    # Home Page
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        st.title("Welcome to the Extraction App")

        col1, col2 = st.columns(2)

        with col1:

            if st.button("URL Extraction",key = "url_tile"):
                st.session_state.page = "url_extraction"

        with col2:
            if st.button("Document Extraction"):
                st.session_state.page = "document_extraction"

    elif st.session_state.page == "url_extraction":
        url_extraction_page()
    elif st.session_state.page == "document_extraction":
        document_extraction_page()

# URL Extraction Page
def url_extraction_page():
    st.title("URL Extraction")

    tab1, tab2, tab3, tab4 = st.tabs(["Extract Links from pdf", "Crawl url and fetch  links", "Fetch Content","Fetch content from deep crawl"])

    with tab1:

        st.header("Extract Links")
        # url = st.text_input("Enter the URL")
        file = st.file_uploader("Upload a file", type="pdf")
        if st.button("Extract"):
            with st.spinner("Extracting Links from Document"):
                st.write(f"Links extracted from {file} (example result)")
                file_bytes = file.read()
                pdf_text = read_pdf(io.BytesIO(file_bytes))
                # st.write(pdf_text)
                links_dict = get_hyper_Links(io.BytesIO(file_bytes))
                st.write("Found Links:")
                st.json(links_dict)

                # Optional: Allow download of scraped links
                df = df = pd.DataFrame(
                                        [(key, link) for key, links in links_dict.items() for link in links],  # Flatten key-value pairs
                                        columns=["Page Number", "Links"]  # Set headers
                                    )
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Links as CSV",
                    data=csv,
                    file_name="scraped_links.csv",
                    mime="text/csv"
                )
    with tab2:
        st.header("crawl url and fetch links")
        url = st.text_input("enter url")
        if st.button("crawl"):
            with st.spinner("Fetching Links"):
                web_links = extract_links(url)
                st.json(web_links)
                df = df = pd.DataFrame(
                    [(key, link) for key, links in web_links.items() for link in links],  # Flatten key-value pairs
                    columns=["Page Number", "Links"]  # Set headers
                )
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Links as CSV",
                    data=csv,
                    file_name="scraped_links.csv",
                    mime="text/csv"
                )

    with tab3:
        st.header("Fetch Content")
        url_to_extract_content = st.text_input("enter url to fetch content")
        if st.button("crawl and get content"):
            with st.spinner("Fetching Content From URL"):
                markdown_content = scrape_and_convert_to_markdown(url_to_extract_content)
                st.markdown(markdown_content)
    with tab4:
        st.header("Fetch Content using Deep Crawl")
        url_to_extract_content = st.text_input("enter url to fetch deep scraped content")
        crawl_depth = st.text_input("enter depth to fetch deep scraped content")
        if st.button("crawl and get deep scrape content"):
            with st.spinner("Crawling multiple layers and fetching content"):
                markdown_content = deep_scrape(url_to_extract_content,int(crawl_depth))
                st.write("scraped")
                for url, content in markdown_content.items():
                    with st.expander(f"Content from {url}"):
                        st.markdown(content)


    if st.button("Back to Home"):
        st.session_state.page = "home"


# Document Extraction Page
def document_extraction_page():
    st.title("Document Extraction")

    tab1, tab2, tab3 = st.tabs(["System Generated Pdfs", "Scanned Pdfs", "Data Processing"])

    with tab1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a document for pdf extraction", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            st.success(f"Uploaded {uploaded_file.name}")
            # readfile(uploaded_file)
        if st.button("Extract from system generated Document"):
            with st.spinner("Extracting Content from the PDF"):
                import pymupdf4llm
                bytes = uploaded_file.read()
                pypdf = pymupdf.open(stream=io.BytesIO(bytes))
                doc = pymupdf4llm.to_markdown(pypdf)
                st.text_area("Extracted PDF Text", doc, height=300)
                st.markdown(doc)
                st.success(f"Uploaded {uploaded_file.name}")
                st.write("Text extraction feature coming soon!")

    with tab2:
        st.header("Extract Text")

        st.title("PDF to LaTeX Converter using Nougat")

        # File uploader for PDF
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file:
            # Save the uploaded file to a temporary location
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            input_path = os.path.join(temp_dir, uploaded_file.name)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File uploaded and saved as: {uploaded_file.name}")

            # Output file path
            output_path = os.path.join(temp_dir, "output.tex")

            # Button to trigger processing
            if st.button("Convert to LaTeX"):
                with st.spinner("Transforming Scanned PDF into MarkDown"):
                    try:
                        cuda_available = torch.cuda.is_available()
                        cuda_device = torch.cuda.current_device() if cuda_available else None
                        st.write(f"CUDA Available: {cuda_available}")
                        if cuda_available:
                            st.write(f"CUDA Device: {torch.cuda.get_device_name(cuda_device)}")
                        # Nougat command
                        command = ["nougat", input_path, "-o", output_path, "--no-skipping", "--recompute"]
                        st.write("started")
                        # Execute the command
                        # result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                                   encoding='utf-8')
                        process.wait()
                        st.success("Conversion successful!")

                        # Display the output
                        with open(output_path + '\\' + uploaded_file.name.split('.')[0] + '.mmd', "rb") as output_file:
                            latex_content = output_file.read().decode('utf-8')

                        st.markdown(latex_content)

                        # Allow the user to download the output file
                        with open(output_path + '\\' + uploaded_file.name.split('.')[0] + '.mmd',
                                  "rb") as file_to_download:
                            st.download_button(
                                label="Download LaTeX file",
                                data=file_to_download,
                                file_name="output.tex",
                                mime="text/plain",
                            )

                    except subprocess.CalledProcessError as e:
                        st.error("An error occurred while converting the file.")
                        st.error(e.stderr)

            st.write("Text extraction feature coming soon!")

    with tab3:
        st.header("Analyze Data")
        st.title("Longformer Summarizer for Large Text")
        st.write("Summarize long documents efficiently using Longformer!")

        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            st.text_area("Original Text", text, height=300)

            model, tokenizer = initialize_model()

            if st.button("Summarize"):
                with st.spinner("Summarizing... This might take a while for very large text!"):
                    summary = summarize_large_text(text, model, tokenizer)
                    st.success("Summarization Complete!")
                    st.text_area("Summary", summary, height=200)


    if st.button("Back to Home"):
            st.session_state.page = "home"

@st.cache_resource
def initialize_model():
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384").to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer
def chunk_text(text, tokenizer, max_length=16384):
    tokens = tokenizer.encode(text, truncation=False)
    for i in range(0, len(tokens), max_length):
        yield tokenizer.decode(tokens[i:i+max_length])


def summarize_large_text(text, model, tokenizer, max_length=16384, summary_max_length=512):
    summaries = []
    for chunk in chunk_text(text, tokenizer, max_length):
        inputs = tokenizer(
            chunk, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=summary_max_length,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            top_p=0.9,
            top_k=50,
        do_sample = True
        )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return " ".join(summaries)
def read_pdf(file):
    # Open the PDF file using fitz
    doc = fitz.open(stream=file, filetype="pdf")
    text = ""
    # Iterate over each page and extract text
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load page
        text += page.get_text()  # Extract text from the page
    return text
def get_link_type(url, base_url):
    """Determines if the link is internal or external."""
    print(url)
    print(base_url)
    base_domain = urlparse(base_url).netloc
    full_url = urljoin(base_url, url)
    link_domain = urlparse(full_url).netloc
    return 'internal' if base_domain == link_domain else 'external'


def extract_links(url):
    """Extracts all internal and external links from the given URL."""
    response = requests.get(url)

    if response.status_code != 200:
        return f"Failed to retrieve content. Status code: {response.status_code}"

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    print("links : ", links)

    link_data = {'internal': [], 'external': []}

    for link in links:
        href = link.get('href')
        text = link.get_text(strip=True)
        title = link.get('title', '')

        link_info = {'href': href, 'text': text, 'title': title}

        link_type = get_link_type(href, url)

        if link_type == 'internal':
            if not href.startswith('http'):
                link_info['href'] = urljoin("https://"+urlparse(url).netloc, link.get('href'))
            link_data['internal'].append(link_info)
        else:
            link_data['external'].append(link_info)

    return link_data


def scrape_and_convert_to_markdown(url):
    # Fetch the webpage content
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve the webpage: {url}")
        return

    html_content = response.text

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Convert the parsed HTML to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False  # Set to True if you want to ignore links
    markdown_content = h.handle(str(soup))

    # Save the markdown content to a file
    # with open(markdown_file_path, 'w', encoding='utf-8') as markdown_file:
    #     markdown_file.write(markdown_content)
    st.download_button(
        label="Download Markdown File",
        data=markdown_content,
        file_name='Scrapedmarkdown',
        mime="text/markdown"
    )
    return markdown_content
    # print(f"Markdown file saved to {markdown_file_path}")

def get_hyper_Links(path):
    """
    param path: path of pdf

    :return: List[String]

    This method will get all the hyper links of the pdf
    """
    page=fitz.open(stream=path, filetype="pdf")
    Links =list()
    final_links = list()
    page_link_dict = dict()
    count = 0
    for i in range(len(page)):
        Links.append([i for i in page.load_page(i).get_links() if len(i) >0])
    for i in Links:
        final_links.append([j["uri"] for j in i if "uri" in j.keys()])
        page_link_dict[count] = [j["uri"] for j in i if "uri" in j.keys()]
        count +=1
    final_links = [i for i in final_links if len(i)>0]
    # print(page_link_dict)
    return page_link_dict


async def multi_level_scrap(url):
    print("enter")
    crawlScrapper = dict()
    async with AsyncWebCrawler(verbose=True) as crawler:

        result = await crawler.arun(url=url,
                                    bypass_cache=True,
                                    )
        internalLinks = result.links.get("internal")
        crawlScrapper[url] = result.markdown
        for i in internalLinks:
            temp = await crawler.arun(url=i.get("href"),
                                    bypass_cache=True,
                                    )
            crawlScrapper[i.get("href")] = temp.markdown

    # crawler = WebCrawler(verbose=True)
    # result = crawler.run(url="https://example.com", bypass_cache=True)
    # internalLinks = result.links.get("internal")
    # for i in internalLinks:
    #     temp = crawler.run(url=i.get("href"),
    #                               bypass_cache=True,
    #                               )
    #     crawlScrapper[i.get("href")] = temp.markdown
    print("exit")
    return crawlScrapper
def deep_scrape(url, extraction_level, current_level=0, visited=None):
    # links = [i.get('href') for i in extract_links(url).get("internal")]
    print(url)
    scraped_data = dict()
    if visited is None:
        visited = set()

        # Avoid visiting the same URL multiple times
    if url in visited:
        return {}

        # Mark this URL as visited
    visited.add(url)


    # Fetch the webpage content
    response = requests.get(url)
    if response.status_code != 200:
        return {url: f"Failed to retrieve content. Status code: {response.status_code}"}

    html_content = response.text

    # Convert the HTML to markdown
    soup = BeautifulSoup(html_content, 'html.parser')
    h = html2text.HTML2Text()
    h.ignore_links = False  # Keep the links in the markdown
    markdown_content = h.handle(str(soup))

    # Add the scraped content for this page
    scraped_data = {url: markdown_content}

    # Base case: Stop recursion if the current level exceeds the extraction level
    if current_level >= extraction_level:
        return scraped_data
    links = soup.find_all('a', href=True)
    print("links : ",links)
    # Find all hyperlinks on the current page
    for link in links:
        if get_link_type(link['href'], url) == 'internal':
            # Resolve relative URLs to absolute URLs
            # Recursively scrape linked pages
            child_data = deep_scrape(urljoin("https://"+urlparse(url).netloc, link.get('href')), extraction_level, current_level + 1, visited)
            scraped_data.update(child_data)

    return scraped_data

if __name__ == "__main__":

    main()
