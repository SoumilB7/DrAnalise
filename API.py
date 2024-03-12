import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
import openai
from scipy.spatial.distance import cosine # pip install scripy
from sentence_transformers import SentenceTransformer, util

#  - - - - 
def dataset_initialize():
    f = open('Ideas.txt','r')
    am = list(f.readlines())
    my_list = [string.rstrip('\n') for string in am]
    f.close()
    return my_list
#  Fetching the dataset

def five_random():
    arr = dataset_initialize()
    ix = []
    for x in range(5):
        ix.append(arr[random.randint(0,len(arr)-1)])
    return ix
#  - - - - 

def appending(reply):
    f = open('Ideas.txt','a')
    f.write(reply+'\n')
    f.close()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
class SimilarityChecker:
    def __init__(self, threshold=0.85, batch_size=32):
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_model = model
        self.data = []
    
    def data_initialize(self):
        self.data = dataset_initialize()  # initializing to the og one 

    def add_data(self, new_ideas):
        self.data.extend(new_ideas)
    
    def check_similarity(self, new_idea):
        new_vector = self.use_model.encode([new_idea], convert_to_tensor=True)
        num_batches = (len(self.data) + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            old_batch = self.data[start_idx:end_idx]
            old_vectors = self.use_model.encode(old_batch, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(new_vector, old_vectors)
            max_similarity_score = similarity_scores.max()
            if max_similarity_score > self.threshold:
                return True, max_similarity_score.item()
        return False, None




def idea_chk(ideam):    # this will check the dataset(txt file) for similar ideas 
    similarity_checker = SimilarityChecker()
    similarity_checker.data_initialize()
    is_similar, similarity_percent = similarity_checker.check_similarity(ideam)
    if not is_similar:
        return True  # idea is new
    else:
        return False # idea is similar




def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None,
    )
    reply = response.choices[0].text.strip()
    return reply

API_key = "" #insert your key 



# ----------------------------------------------
def API_call_gen():
    Idea_gen_prompt = '''Imagine you are an entrepreneurial thought leader skilled at conceptualizing transformative startup ideas. I will provide 5 example ideas from the Tech domain. Use these as inspiration to propose 12 highly original ideas that solve important problems through unconventional, potentially disruptive approaches. 
        The new ideas should:
        - Explore truly novel applications of emerging technologies like AI, robotics, blockchain, VR/AR, , synthetic biology, etc. integrated in a purposeful way
        - Go beyond incremental improvements - challenge assumptions and standard practices in the industry
        - Consider specialized business applications and scientific use cases, not just consumer products 
        - Combine interdisciplinary thinking and lateral connections between different fields
        - include in near to low context of the ideas I provide .
        - all the ideas should have consistent commercial viability while maintaining the same creativity

        While the examples demonstrate the quality and logical coherence expected, do not limit yourself to their domains or approaches. The goal is to generate innovative ideas that push boundaries and represent fundamentally new ways of thinking about problems and opportunities in Tech.

        Example ideas:

        1. {}
        2. {}
        3. {}
        4. {}
        5. {}

        Innovative new ideas for Tech :

        1.
        2. 
        3.
        4. 
        5.
        6. 
        7.  
        8.
        9.  
        10.
        11.
        12.

        The output should be of the JSON format:
        {
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
            <Theme> : <Idea>;
        }
    ''' 
    idear = five_random()
    Idea_gen_prompt.format(idear[0],idear[1],idear[2],idear[3],idear[4])
    Prompt = '\nUser: ' +  Idea_gen_prompt
    reply = chat_with_gpt(Prompt)
    return reply # reply
    
def API_call_ana(ide):
    Idea_ana_prompt = '''
    Please analyze the following startup idea in the technology space. 
    Provide:
        - A brief 2-3 sentence overview of the central concept and value proposition
        - An implementation walkthrough explaining how this could be built and launched in a simplified, general way. Use 5-7 bullet points to map out key stages and steps. 
        - A list of 5 technical roles or skills that would likely be necessary to execute this idea and bring it to market. Use bullet points.
        - A pros and cons list with 3-5 high-level benefits and risks of this concept. Use bullet points.
    
    When discussing implementation, aim for a broad beginner-level understanding, not overly detailed. Provide enough info to convey the core technologies and processes needed. 
    The goal is to provide an overview of the key aspects of this idea to evaluate its viability and support initial planning. Let me know if you would like me to modify or expand this prompt in any way. Otherwise I believe we have a solid framework here to analyze a startup idea from multiple angles.
    Startup Idea: ```{}```
    Analysis:


    '''.format(ide)
    Prompt = '\nUser: ' +  Idea_ana_prompt
    reply = chat_with_gpt(Prompt)
    return reply


def API_personalized_gen(Stack : str):
    Idea_Prsn_prompt = '''Please read the following user's background and interests paragraph:

        ```{}```

        Based on their skills, passions, and goals described above, please propose 5 logical and feasible startup ideas that align with their abilities and interests. 

        When generating the ideas:
        - The ideas should have real world application and act as a business owner to access the ideas and only allow the ideas to be outputted as long as they have a real world application

        - Feel free to draw from diverse industries and domains, not just the explicitly stated field. Explore areas related to their interests more broadly.

        -Strictly replay with ideas that are feasible as an early-stage startup with suitable base of how they can launch and incrementally scale rather than require massive upfront development.

        - Provide examples which have novel ways to integrate the user's skills and interests in unexpected combinations.

        - Consider leveraging their capabilities in tangent industries or new applications of underlying technologies.

        - Suggest ideas that combine the user's interests with solving problems in completely different domains.

        - Think on ::  how would the idea differentiate itself from others in the space? What would the competitive advantage be?

        - Introduce constraints or requirements that would lead ideas in new directions.

        - The ideas should be very niche focused and should not have a broad range of work.

        For each startup idea: 

        - Provide 2-3 sentences describing the core business concept and value proposition.

        The goal is to provide realistic yet original startup possibilities tailored to this individual's skills and passions. Please give the ideas in a json format:

        {
        [Idea theme] : [Explanation in 2-3 lines],
        [Idea theme] : [Explanation in 2-3 lines],
        [Idea theme] : [Explanation in 2-3 lines],
        [Idea theme] : [Explanation in 2-3 lines],
        [Idea theme] : [Explanation in 2-3 lines],
        }
    '''.format(Stack)
    Prompt = '\nUser: ' +  Idea_Prsn_prompt
    reply = chat_with_gpt(Prompt)
    return reply

# -----------------------------------------------   
#  API Call established

def idea_append():
    gpt_reply = API_call_gen()
    list_ideas = list(gpt_reply.split('\n'))
    for id in list_ideas:
        if idea_chk(id):  # checking idea 
            appending(id)  # appending the idea 



app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def index():
    return render_template('index.html', items = "func")
    
@app.route('/process_data', methods=['POST'])
def process_data():
    data = json.loads(request.data)
    input_value = data['input']
    #------------------------------------------------------------
    # based on the input value

    # authorization for a task : 
    if input_value == "idea gen":
        idea_append()
        # format of input_value : "idea gen"

    if "analysis : " in input_value:
        output_ana : API_call_ana(input_value.removeprefix("analysis : "))
        # format of input_value : "analysis :" + idea 

    #------------------------------------------------------------
    return jsonify({'output': output_ana})

if __name__ == '__main__':
    
    app.run(port='5000',debug=True)