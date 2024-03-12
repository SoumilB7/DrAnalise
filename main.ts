
import { Configuration, OpenAIApi } from "openai";

const configuration = new Configuration({
  apiKey: "",
});

const openai = new OpenAIApi(configuration);

async function generateText(prompt: string, maxTokens: number) {
  const response = await openai.createCompletion({
    model: "text-davinci-003", 
    prompt: prompt,
    max_tokens: maxTokens,
  });

  return response.data.choices[0].text;
}



// --------------------------------- Prompts --------------------------------

async function Idea_prsn(paragraph: string,skills:Array<string>)
{
  if(paragraph.length==0){

  }
  const Idea_prsn_prompt = `
    Based on their skills, passions, and goals described in the user info below, please propose 5 logical and feasible startup ideas that align with their abilities and interests. 
    User info :
      about : '''${paragraph}'''
      Skills : '''${skills}'''
    When generating the ideas:
    - The ideas should have real world application and act as a business owner to access the ideas and only allow the ideas to be outputted as long as they have a real world application

    - Feel free to draw from diverse industries and domains, not just the explicitly stated field. Explore areas related to their interests more broadly.

    - Strictly replay with ideas that are feasible as an early-stage startup with suitable base of how they can launch and incrementally scale rather than require massive upfront development.

    - Provide examples which have novel ways to integrate the user's skills and interests in unexpected combinations.

    - Consider leveraging their capabilities in tangent industries or new applications of underlying technologies.

    - Suggest ideas that combine the user's interests with solving problems in completely different domains.

    - Think on how would the idea differentiate itself from others in the space? What would the competitive advantage be?

    - Introduce constraints or requirements that would lead ideas in new directions.

    - The ideas should be very niche focused and should not have a broad range of work.

    - you can also include the ideas which need certain skills and might require more skills so that we could get the best ideas there are. 

    For each startup idea: 
    - Provide 2-3 sentences describing the core business concept and value proposition.


    The goal is to provide realistic yet original startup possibilities tailored to this individual's skills and passions. 

  Please give the ideas only in a json format :
    {
    {
      [Idea theme] : [Explanation in 2-3 lines],
      [skills matched] : [skills that are in the idea's requirements]
    },
    {
      [Idea theme] : [Explanation in 2-3 lines],
      [skills matched] : [skills that are in the idea's requirements]
    },
    {
      [Idea theme] : [Explanation in 2-3 lines],
      [skills matched] : [skills that are in the idea's requirements]
    },
    {
      [Idea theme] : [Explanation in 2-3 lines],
      [skills matched] : [skills that are in  the idea's requirements]
    },
    {
      [Idea theme] : [Explanation in 2-3 lines],
      [skills matched] : [skills that are in the idea's requirements]
    }
    }
    `
    const text = await generateText(Idea_prsn_prompt, 2000); // API call
    // console.log(text);
    return text;
};   // it will return JSON for the set of 5 personalized idea

async function Idea_ana(idea:string)
{
    const Idea_ana_prompt =`
    Please analyze the following startup idea in the technology space. 
    Provide:
        - A brief overview of the central concept and value proposition and with great details on the startup explanation and central idea
        - A product description of what exactly is the product is and what is that the market needs the product to have for it to acquire the market , make everything into a paragraph
        - An implementation walkthrough explaining how this could be built and launched in a simplified, general way. Use 4 bullet points to map out key stages and easy to follow steps. 
        - A list of 5 technical roles or skills that would likely be necessary to execute this idea and bring it to market. Use bullet points.
        - A pros and cons list with 3-5 high-level benefits and risks of this concept. Use bullet points.
        - A market analysis of all the prerequisites that are needed to make the idea and please provide under what specific conditions and financial requirements can this idea turn into a profitable business, make everything into a paragraph (not points)

    
    When discussing implementation, aim for a broad beginner-level understanding, not overly detailed. Provide enough info to convey the core technologies and processes needed. 
    The goal is to provide an overview of the key aspects of this idea to evaluate its viability and support initial planning. Let me know if you would like me to modify or expand this prompt in any way. Otherwise I believe we have a solid framework here to analyze a startup idea from multiple angles.
    Startup Idea: '''${idea}'''
    Analysis:
    Please provide the output only in the following JSON format (no extra output) :
    {
      Overview : "  -  " ,
      Product_description : " - ",
      Implementation : ["step 1","step 2","step 3", ....... ,"step n" ],
      Tech_Roles : ["role 1","role 2","role 3", ....... , "role n"],
      Market_Analysis : " -- ",
      Pros : ["pro 1","pro 2",....."pro n"],
      Cons : ["con 1","con 2", .... ,"con n"]
    }
    ` 
    const text = await generateText(Idea_ana_prompt, 2000);  // API call
    // console.log(text);
    return text;
};   // it will return a big string of deep analysis of that idea 

// --------------------------------- Prompts --------------------------------


console.log(Idea_ana("Augmented reality (AR) shopping app that allows users to virtually try on clothes"))

// The way to connect with the front end 
