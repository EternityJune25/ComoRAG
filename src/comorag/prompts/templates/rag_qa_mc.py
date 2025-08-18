one_shot_rag_qa_docs = (
    """Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

# one_shot_ircot_demo = (
#     f'{one_shot_rag_qa_docs}'
#     '\n\nQuestion: '
#     f"When was Neville A. Stanton's employer founded?"
#     '\nOptions: '
#     f"A) 1862\nB) 1950\nC) 1952\nD) 2010"
#     '\nThought: '
#     f"The employer of Neville A. Stanton is University of Southampton. According to the passage about Southampton, the University of Southampton was founded in 1862. So the answer is [A]."
#     '\n\n'
# )

# rag_qa_system = (
#     'You are an expert on reading and understanding books and articles. Given the following excerpt from an article, and a related question with different options, you need to analyze which option is the best answer for the question.\n\n【article】\n{context}\n\n【question】\n{question}\n\n【options】\n{options}\n\nPlease try your best to pick the correct option; if you are not sure of the answer, then pick your best guess. For your response, just pick the best option in the end, in the format of [A] or [B] or [C] or [D], do not explain why'
# )

#不要删除！！！！！！
rag_qa_system = (
    '''
    ### Role
    You are an expert on reading and understanding books and articles.
    ### Task 
    Given the following detail article, summary by semantic and summary by timeline from a book, and a related question with different options, you need to analyze which option is the best answer for the question.
    
    ### Detail Article
    {context}

    ### Summary by Semantic
    {semantic_summary}
    
    ### Summary by Timeline
    {timeline_summary}

    ### question
    {question}

    Please try your best to pick the correct option; 

    ### Limits
    1. Do not infer, response only based on the content strictly !!!
    2. Pick the choice only if you find at least 2 places that support the answer !!!
    
    ### Response Format
    1. Start with a brief summary of the content in no more than three sentences. Begin this section with "### Content Understanding"
    2. Based on the question, analyse and list all relevant items using a markdown list. Begin this section with "### Question Analyse".
    3. Extract the key points related to 4 options, also using a markdown list. Begin this section with "### Options analyse". Note: Only analyze based on the provided materials, do not make guesses.
    4. Provide your final answer with a heading. Begin this section with "### Final Answer" followed by the best option in the format of [A] or [B] or [C] or [D] without explaining why.
    '''
)


# rag_qa_system = (
#     '''
#     ### Role
#     You are a sophisticated reading comprehension expert specializing in multi-source text analysis. Your expertise includes:
#     - Synthesizing information from different text representations (detailed articles, semantic summaries, and timeline summaries)
#     - Identifying relevant evidence across multiple document types
#     - Making evidence-based inferences while maintaining objectivity
#     - Avoiding speculation beyond the provided materials
    
#     ### Task 
#     Your task is to solve a multiple-choice question by analyzing information from three different sources about a book. You will:
#     1. Examine the detailed article, semantic summary, and timeline summary provided
#     2. Identify information relevant to the question across all three sources
#     3. Evaluate each answer option against the available evidence
#     4. Select the best answer based solely on the provided materials
#     5. If insufficient information exists to determine the answer, acknowledge this limitation
    
#     ### Detail Article
#     {context}

#     ### Summary by Semantic
#     {semantic_summary}
    
#     ### Summary by Timeline
#     {timeline_summary}

#     ### question
#     {question}

#     ### Response Format
#     1. Start with a brief summary of the content in no more than three sentences. Begin this section with "### Content Understanding"
#     2. Based on the question, analyse and list all relevant items using a markdown list. Begin this section with "### Question Analyse".
#     3. Extract the key points related to 4 options, also using a markdown list. Begin this section with "### Options analyse". Note: Only analyze based on the provided materials, do not make guesses.
#     4. Provide your final answer with a heading. Begin this section with "### Final Answer" followed by the best option in the format of [A] or [B] or [C] or [D] without explaining why; if you cannot answer, only output a "*".
#     '''
# )

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    "A) 1862\nB) 1950\nC) 1952\nD) 2010"
    
)

one_shot_rag_qa_output = (
    "### Content Understanding\n"
    "The text discusses Neville A. Stanton who is a professor at the University of Southampton. The University of Southampton is described as a public research university in Southampton, England that was founded in 1862 as the Hartley Institution.\n\n"
    
    "### Question Analyse\n"
    "- The question asks about when Neville A. Stanton's employer was founded\n"
    "- Neville A. Stanton's employer is identified as the University of Southampton\n"
    "- The founding date of the University of Southampton needs to be determined\n\n"
    
    "### Options Analyse\n"
    "- A) 1862: The text states that the University of Southampton was founded in 1862 as the Hartley Institution\n"
    "- B) 1950: This date is not mentioned in the text in relation to the University of Southampton's founding\n"
    "- C) 1952: This date is not mentioned in the text in relation to the University of Southampton's founding\n"
    "- D) 2010: This date is not mentioned in the text in relation to the University of Southampton's founding\n\n"
    
    "### Final Answer\n"
    "[A]"
)

prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user} "}
]