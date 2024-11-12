from fasthtml_hf import setup_hf_backup
import io
import os
import traceback
from pydantic_core import from_json
from fasthtml.common import * 
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser

# Initialize the fastHtml application
app, rt = fast_app()

# Define Pydantic models for structured output

# SummaryLine represents a single summary item with its keywords and description
class SummaryLine(BaseModel):
    summary_item: str = Field(description = "Actual summary sentence that contains highlighting key data points or information.", 
                              max_length = 200)
    keywords: List[str] = Field(description = "A list of exact words or phrases in the summary item that highlights most important data points or key ideas.")
    brief_descripton_of_summary: str = Field(description = "This is elaborate description to provide context or background to the summary item.",
                                              max_length = 500)

# TopicSummaries represents a collection of summaries for a specific topic
class TopicSummaries(BaseModel):
    topic: str = Field(description = "Topics of summary as mentioned in the instructions.")
    summaries: List[SummaryLine] = Field(description = "This a list summary for a topic with each one having it's own keywords and context.",
                                         min_items=3, 
                                         max_items=5)

# CompleteSummary is the top-level model containing all topic summaries
class CompleteSummary(BaseModel):
    summaries_list: List[TopicSummaries]           

# Define the template for summarization
# This template provides instructions to the AI model on how to structure the summary
summarize_template = """
Write a concise summary of the case study given in the context. The summary should be based on the following topics.
"""

# Define the specific sections to be included in the summary
summary_sections = """
- Factual: Facts or information that contains numbers, dates, events etc. that are mostly quantitative or qualitative data
- SWOT: Key Strength, weakness, opportunities or threats that are mentioned in the case study
- Decisions and Outcomes: Key decisions taken and it's successful or failed outcomes and reasons
- Ethical and Governance: Key considerations from ethical and governance perspective

"""

# Define the context string for one-pass summarization
# This string provides additional formatting instructions for the summary
context_str = """
<context>
{context_content}
</context>

The response must follow the following schema strictly. There will be penalty for not following the schema.
"""

# Define the template for the reduce step in map-reduce summarization
# This template instructs the model to consolidate multiple summaries into a final summary
refine_str = """The following are set of summaries given in a markdown format:

{previous_summary}

Now add the above summary with more context given below and create final summary, which should contain the following sections.
"""

# Function to get the appropriate language model based on user selection
def getModel(model, key):
    if(model == 'OpenAI'):
        os.environ['OPENAI_API_KEY'] = key
        return ChatOpenAI(temperature=0,  # Set to 0 for deterministic output
                    model="gpt-4o",  # Using the GPT-4 Turbo model
                    max_tokens=4096)  # Limit the response length
    elif (model == 'Anthropic'):
        os.environ['ANTHROPIC_API_KEY'] = key
        return ChatAnthropic(model='claude-3-5-sonnet-20240620')  # Limit the response length
    else:    
        return OllamaLLM(model="gemma:2b")
    
# Function to highlight specific keywords in the text
def highlight_text(text, key_words):
    for word in key_words:
        text = text.replace(word, f'<span style="color:red;"><b>{word}</b></span>')    
    html_text = "<div>" + text + "</div>"
    return eval(html2ft(html_text))

# Function to generate an HTML table from the summary object
def generate_table(summaries_obj):
    column_names = ['Topic', "Summary"]
    table_header = Thead(Tr(*[Th(key) for key in column_names]))
    table_rows = []
    for topic_summary in summaries_obj.summaries_list:            
        first_row = True
        for summary in topic_summary.summaries:
            if(first_row):
                table_rows.append(Tr(Td(topic_summary.topic,
                                        rowspan=f"{len(topic_summary.summaries)}", 
                                        style = "width: 10%;"), 
                                     Td(highlight_text(summary.summary_item, summary.keywords), 
                                        style = "width: 60%;"),
                                     Td(Div(Details(Summary( style = "summary::-webkit-details-marker { display: none }; list-style-type: '+'"), 
                                                            P(summary.brief_descripton_of_summary)),
                                            style ="padding: 0.5em 0.5em 0;"),
                                            style = "width: 30%;")))
                first_row = False
            else:
                table_rows.append(Tr(Td(highlight_text(summary.summary_item, summary.keywords), 
                                        style = f"width: 60%; rowspan='{len(topic_summary.summaries)}'"),
                                     Td(Div(Details(Summary( style = "summary::-webkit-details-marker { display: none }; list-style-type: '+'"), 
                                                               P(summary.brief_descripton_of_summary)), 
                                            style ="padding: 0.5em 0.5em 0;"),
                                            style = "width: 30%;")))                

    return Div(Card(Table(table_header, Tbody(*table_rows))))

# Function to perform one-pass summarization on the given pages
def onepass_summarize(pages, summary_sections, model):
    """
    Perform one-pass summarization on the given pages.
    
    This function creates a summarization chain using the provided instructions
    and model, then applies it to the input pages to generate a summary.
    
    Args:
    pages (list): List of pages (documents) to summarize
    instructions (str): Custom instructions for summarization
    model (ChatOpenAI): Instance of ChatOpenAI model to use for summarization
    
    Returns:
    str: Summarized text in markdown format
    """
    onepass_summary_template = summarize_template + summary_sections + context_str + "{format_instructions}"
    print("Onepass instruction: " + onepass_summary_template)

    output_parser = PydanticOutputParser(pydantic_object=CompleteSummary)
    format_instructions = output_parser.get_format_instructions()
    print("Format instructions: " + format_instructions)

    # Create a prompt template combining the instructions and context
    prompt = PromptTemplate.from_template(onepass_summary_template)
    # Create an LLM chain with the model and prompt
    summary_chain = prompt | model | output_parser

    print("Getting Summary......")
    # Invoke the chain on the input pages and return the summarized text
    summaries = summary_chain.invoke({"context_content": pages, 
                                   "format_instructions": format_instructions})
    return summaries
    
# Function to generate the configuration form for the web interface
def getConfigForm():
    return Card(Form(hx_post="/submit", hx_target="#result", hx_swap_oob="innerHTML", hx_indicator="#indicator")(
            Div(
                Label(Strong("Model and Prompt Instruction: "), style="color:#3498db; font-size:25px;")
            ),
            Div(
                Label(Strong('Model: ')),
                Select(Option("OpenAI"), Option("Anthropic"), Option("Ollama-Llama3.2"), id="model")
            ),
            Div(
                Label(Strong('Secret Key: ')),
                Input(id="secret", type="password", placeholder="Key: "),
            ),
            Div(
                Label(Strong('Upload File: '), "Upload only pdf file with max size of 1 MB"),
                Input(id="file", type = 'file', placeholder="Key: ", accept = ".pdf", max = '1024000'),
            ),
            Div(
                Label(Strong('Instruction: ')),
                P('Provide the list of topics and their one line description for summarization as shown in example. Summarization will have these sections.', 
                  style = 'font-size: 12px;'),
                Textarea(summary_sections, id="instruction", 
                         style="height:250px")
            ),
            Div(                
                Button("Summarize")
            ),
            Div(              
                Br(),                  
                A("Developed by Manaranjan Pradhan", href="http://www.manaranjanp.com/", 
                  target="_blank", 
                  style = 'color: red; font-size: 16px;')                      
            )))

# Define the route for the homepage
@app.get('/')
def homepage():
    return Titled('Document Summarization', Grid( getConfigForm(),
        Div(
            Div(Label(Strong('Summarizing the document.... take a deep breath....')),
            Progress(), id="indicator", cls="htmx-indicator"),
            Div(id="result", style ="font-family:Helvetica; font-size=24pt;")
        )
        , style="grid-template-columns: 400px 1000px; gap: 50px;"
    ))

# Define the route for form submission
@app.post('/submit')
async def post(d:dict):
    try:
        # Check if a file was uploaded
        if "file" in d.keys():
            pages = await d['file'].read(-1)
            pdf_reader = PdfReader(io.BytesIO(pages))
        else:
            return Div("File not uploaded.", cls = 'alert', )    
            
        # Extract text from each page of the PDF
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"

        # Get the appropriate language model
        model = getModel(d['model'], d['secret'])    
        
        # Perform one-pass summarization
        summaries = onepass_summarize(text_content, d['instruction'], model)

        print(f"Summary Obtained: {summaries}")
        
        # Generate and return the HTML table with the summaries
        return generate_table(summaries)

    except BaseException as e:
        print(traceback.format_exc())
        return str(e)

#setup_hf_backup(app)

# Start the FastAPI server
serve()
