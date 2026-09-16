PROMPTS = {
   "formality": {
      "p1": """You are an AI assistant skilled at transforming formal queries into casual, everyday language. 
      Rewrite the following query so that it sounds very informal. Experiment with different colloquial openings, 
      varied sentence constructions, and a mix of slang, idioms, and casual expressions throughout the sentence. 
      Avoid using the same phrase repeatedly (e.g., 'hey, so like') and ensure the meaning remains unchanged.""",

      "p2": """Your task is to convert the given query into an informal version that feels natural and conversational. 
      Instead of a uniform introductory phrase, use a range of informal expressions (such as interjections, casual 
      questions, or slang) at different parts of the sentence. Mix up the structure—sometimes start with an interjection, 
      other times rephrase the sentence completely—while keeping the original meaning intact.""",

      "p3": """### Task: Transform Formal to Extremely Informal Language

      Convert the following formal sentence into an extremely informal, messy, and natural version. 
      The output should sound like authentic, real-world casual speech—as if spoken in an informal chat, online conversation, or street talk.

      #### Critical Rules:
      ✅ Diversity is key: No two sentences should follow the same pattern.
      ✅ Break formal sentence structures: Chop up long phrases, reorder words, or make them flow casually.
      ✅ Use a wide range of informality techniques—DO NOT rely on just contractions or slang.
      ✅ Avoid starting every sentence with the same phrase or discourse marker.
      ✅ Embrace randomness: Let the outputs sound wild, real, and unpredictable.
      ✅ Do not start your sentences with Yo, Hey so like etc. Make sure you make it sound very informal without using these phrases AT ALL.

      **Final Execution Instruction:**
      Generate an informal version of the following sentence that:
      ✔ Uses multiple different informality techniques.
      ✔ Avoids repetitive sentence structures or patterns.
      ✔ Makes it sound raw, conversational, and unpredictable.

      **Original Query:**""",

       
       
   },

   "readability": {
   'p1': '''1. Task Definition:
You are rewriting a query to make it significantly less readable while preserving the original semantic meaning as closely as possible.

2. Constraints & Goals:

- Flesch Reading Ease Score: The rewritten text must have a Flesch score below 60 (preferably below 50).
- Semantic Similarity: The rewritten text must have SBERT similarity > 0.7 compared with the original query.
- Length: The rewritten text must remain approximately the same length as the original query (±10%).
- Preserve Domain Terminology: Do not remove or drastically change domain-specific words, abbreviations, or technical terms (e.g., "IRS," "distance," etc.). 
- Abbreviation: Do not expand abbreviations unless the original query already used the expanded form.
- No New Information: You must not add additional details beyond what the original query states.
- Question Format: Retain the form of a question if the original is posed as a question.

3. How to Increase Complexity:

- Lexical Changes: Use advanced or academic synonyms only for common words. For domain or key terms (e.g., "distance," "IRS," "tax"), keep the original term or use a very close synonym if necessary to maintain meaning.
- Syntactic Complexity: Introduce passive voice, nominalizations, embedded clauses, and parenthetical or subordinate phrases. Ensure the sentence flow is more formal and convoluted without changing the core meaning.
- Redundancy & Formality: Employ circumlocution and excessively formal expressions (e.g., "due to the fact that" instead of "because") while avoiding any semantic drift.
- Dense, Indirect Construction: Favor longer phrases, indirect references, and wordiness. Avoid direct or simple phrasing.''',

   'p2': '''### **Task Description:**  
Transform a given query into a significantly less readable version while preserving its original semantic meaning as closely as possible.

### **Constraints & Goals:**  
- **Readability:** The rewritten text must have a **Flesch Reading Ease Score below 60**, preferably below 50.  
- **Semantic Similarity:** The rewritten text must achieve an **SBERT similarity score > 0.7** with the original query.  
- **Length Consistency:** The modified text should be **within ±10% of the original length**.  
- **Preserve Key Terminology:** **Do not alter domain-specific words, abbreviations, or technical jargon** (e.g., "IRS," "distance").  
- **Abbreviation Handling:** **Do not expand abbreviations** unless they are already expanded in the original query.  
- **Maintain Original Intent:** Do not add, remove, or alter the factual content of the query.  
- **Retain Question Structure:** If the input is a question, the output must also be a question.  

### **Techniques to Decrease Readability:**  
1. **Lexical Complexity:** Replace common words with **advanced, academic, or formal synonyms**, while keeping domain-specific terms unchanged.  
2. **Syntactic Complexity:** Introduce **passive voice, nominalizations, embedded clauses, or subordinate structures** to increase sentence density.  
3. **Redundancy & Formality:** Use **circumlocution, excessive formality, and indirect phrasing** (e.g., "in light of the fact that" instead of "because").  
4. **Dense Sentence Structure:** Prefer **wordy, indirect, and convoluted constructions** over direct phrasing.''',

   'p3': '''### **Objective:**  
You are tasked with **rewriting a given query to make it significantly less readable** while preserving its original semantic meaning with high fidelity.

### **Guiding Principles:**  
- **Readability Constraint:** The rewritten text must have a **Flesch Reading Ease Score of ≤60**, preferably ≤50.  
- **Semantic Integrity:** Ensure an **SBERT similarity score of at least 0.7** between the original and rewritten text.  
- **Length Tolerance:** Maintain an **approximate length deviation of no more than ±10%** from the original.  
- **Terminology Preservation:** Domain-specific terms (e.g., "IRS," "distance") **must remain intact** or be substituted only with **near-synonymous equivalents**.  
- **Abbreviation Handling:** If an abbreviation exists, **retain it as is** unless the original query explicitly expands it.  
- **Strict Content Preservation:** Do **not introduce any new information** or omit existing details.  
- **Question Retention:** If the input is a question, the reformulated output **must remain a question**.  

### **Techniques for Readability Reduction:**  
- **Lexical Sophistication:** Replace commonplace words with **more complex, formal, or technical alternatives** while maintaining clarity of meaning.  
- **Structural Density:** Employ **passive constructions, embedded clauses, and nominalized phrases** to increase syntactic complexity.  
- **Circumlocution & Wordiness:** Favor **verbose, indirect expressions** over concise phrasing (e.g., "with regard to" instead of "about").  
- **Elaborate Phrasing:** Use **multi-clause structures and intricate sentence formations** to reduce direct readability.''',
},

   "politeness": {
      'p1': '''### Task: Rewrite Queries to Sound More Polite and Courteous  

Rephrase the given query into a more polite, respectful, and considerate version while preserving its original intent. The output should reflect a natural, well-mannered tone suitable for professional or friendly interactions. The generated query should be a single sentence.

#### Critical Rules:  
- Use a variety of politeness techniques, including warm greetings, indirect requests, and expressions of gratitude.  
- Avoid robotic or overly formal constructions—make it sound naturally courteous, warm and friendly.  
- Do not always start your sentence with 'Could you please tell'. Use emotional undertones and specific attempts at politeness.
- Maintain the original meaning without unnecessary embellishment.
- Do not start the generated query with 'I hope you are ...' or end with a single 'Thank you' sentence. Generate only a single polite query sentence.
''',

    'p2': '''### Task: Enhance the Courtesy of a Given Query
    
Transform the provided query into a more respectful, friendly, and warm version, ensuring it conveys respect and warmth while keeping the original intent intact. The reworded request should sound engaging, professional, and well-mannered. The generated query should be a single sentence.

### Key Considerations:
- Use a mix of politeness techniques, including indirect phrasing, friendly introductions, and appreciative language.
- Keep the tone natural—avoid overly rigid or formal wording that feels robotic.
- Vary sentence structures instead of defaulting to "Could you please...". Use emotional undertones and specific attempts at politeness.
- Maintain the original meaning while subtly enhancing the request's politeness and friendliness.
- Avoid beginning the generated query with 'I hope you are...' or concluding it with a separate 'Thank you.' sentence. Generate only one polite query sentence.
''',

    'p3': '''### Task: Refining Queries for Politeness and Warmth

Transform a given query into a more courteous, engaging, and warm request while ensuring it retains the original intent. The revised version should sound friendly, professional, and respectful. The generated query should be a single sentence.

### Guidelines:
- Incorporate politeness techniques such as indirect requests, warm introductions, and appreciative language.
- Ensure the tone is natural—avoid excessive formality that feels robotic.
- Diversify sentence structures rather than defaulting to "Could you please...". Use emotional undertones and specific attempts at politeness.
- Subtly enhance warmth and professionalism while preserving clarity and intent.
- Avoid beginning the generated query with 'I hope you are ...' or concluding it with a standalone 'Thank you' sentence. Generate only one polite query sentence.
'''
},
    
    "sentence_type":{
        'p1': '''### Task: Rewrite Query as {sentence_type} Sentence
Take the given query and rewrite it as a {sentence_type} sentence, changing only the sentence type while keeping the original meaning intact. 
Do not add extra details or change the intent.'''
    },
    
"grammar_error": {
      "p1": """You are an AI assistant skilled at generating realistic grammatical errors in queries.
Rewrite the given query by introducing grammatical errors.
Constraints:
- Preserve the original meaning
- Introduce at least three grammatical errors per query
- Make it realistic and natural by varying the types of errors """
       
       
   },


}