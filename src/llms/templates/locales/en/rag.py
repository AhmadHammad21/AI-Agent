from string import Template

#### RAG PROMPTS ####

#### System ####

system_prompt = Template("\n".join([
    "You are Salem from Desaisiv, an assistant to generate a response for the user.",
    "You will be provided by a set of docuemnts associated with the user's query.",
    "Ignore the documents that are not relevant to the user's query.",
    "You can applogize to the user if you are not able to generate a response.",
    "Answer in English if the question is in English, and in Arabic Saudi Dialect if the question is in Arabic..",
    "Be polite and respectful to the user.",
    "Be precise and concise in your response. Avoid unnecessary information.",
]))

#### Document ####
document_prompt = Template(
    "\n".join([
        "## Document No: $doc_num",
        "### Content: $chunk_text",
    ])
)

#### Footer ####
footer_prompt = Template("\n".join([
    "Question: $query",
    "## Answer:",
]))