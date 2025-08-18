

one_shot_rag_qa_docs = (
    """Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)


rag_qa_system = (
    '''
    ###Role
    You are an expert at carefully reading complex texts, extracting narrative details, and making logical inferences.
    
    ###Task
    Given the following detail article from a book, and a related question, you need to provide a comprehensive and accurate answer based on the given information.
    
    ### Detail Article
    {context}

    ### question
    {question}

    ###Response Format
    1. Start with a brief understanding of the content in no more than three sentences. Begin this section with "### Content Understanding"
    2. Based on the question, identify and analyze all information in the content that could plausibly be relevant. Be thorough, but do not add anything not supported by the text. Use a markdown list. Begin this section with "### Relevant Information Analysis"
    3. From the relevant information, extract and organize only the key facts that directly support answering the question. Use a concise markdown list. Begin this section with "### Key Facts"
    4. Add your final answer at the end in the format "### Final Answer." Provide the shortest possible answer that is supported by the text. If there isn’t enough information in the content to provide a confident or definite answer, simply write "*".
    '''
)
rag_qa_system = (
    '''
    ### Role
    You are an expert at carefully reading complex texts, extracting narrative details, and making logical inferences.
    
    ### Task
    Given the following detail article from a book, and a related question, you need to provide a accurate answer based on the given information.Use the shortest possible answer taken from the text. 
    
    ### Detail Article
    {context}

    ### question
    {question}

    ### Response Format
    0. All numbers must be written in English words for example twenty-three instead of twenty-three. Do not output approximations inequalities or ranges Give an exact answer from the text if available
    1. Start with a very brief understanding of the content in no more than two sentences. Begin this section with "### Content Understanding"
    2. Identify and analyze all plausibly relevant information from the content. Use a short markdown list. Avoid adding anything not in the text. Begin this section with "### Relevant Information Analysis"
    3. From that, extract only the key facts that directly answer the question. Use a concise markdown list. Begin this section with "### Key Facts"
    4. Add your final answer in the format "### Final Answer." Use the shortest possible answer taken from the text. If there isn’t enough information, just write "*"
    '''
)


one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
)


one_shot_rag_qa_output = (
    "### Content Understanding\n"
    "The text discusses Neville A. Stanton who is a professor at the University of Southampton. The University of Southampton is described as a research university that was founded in 1862 and received its Royal Charter as a university in 1952. The university has over 22,000 students and is ranked among the top research universities globally.\n\n"
    
    "### Relevant Information Analysis\n"
    "- Neville A. Stanton is identified as a Professor at the University of Southampton\n"
    "- The University of Southampton is his employer\n"
    "- The text provides founding information for the University of Southampton\n"
    "- The university was founded in 1862\n"
    "- It received its Royal Charter as a university in 1952\n\n"
    
    "### Key Facts\n"
    "- Neville A. Stanton's employer: University of Southampton\n"
    "- University of Southampton founding date: 1862\n"
    "- University of Southampton received Royal Charter: 1952\n\n"
    
    "### Final Answer\n"
    "1862."
)

prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]