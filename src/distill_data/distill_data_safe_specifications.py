# uses the ift dataset and the model mistralai/Mistral-7B-Instruct-v0.2 to generate prompts
# uses a 8-shot prompt to generate them


import json
from tqdm import tqdm
from openai import OpenAI
import os
import time

GPT4 = ''

def getResponse(prompt, engine):
    msg = []
    msg.append({"role": "user", "content": prompt})
 
    client = OpenAI(api_key=GPT4, timeout=300)
    
    output = None
    while output is None:
    # while output is None and times <= 3:
        try:
            response = client.chat.completions.create(
                model=engine,
                max_tokens=2048,
                messages=msg,
                temperature=0.6
                )
            # print(response)
            output = response.choices[0].message.content
        except Exception as e:
            print(e)
            print('Retrying...')
            time.sleep(5)
    return output
    
def main():
    data = json.load(open('','r',encoding='utf-8'))
    result = []
    save_path = ''
    if os.path.exists(save_path):
        result = json.load(open(save_path,'r'))
    format_Str = "<Analysis>[<Step>...</Step><Step>...</Step><Step>...</Step><Step>Requests Categorization...</Step>]</Analysis><Final Answer>...</Final Answer>"
    for line in tqdm(data[len(result):]):
        line_prompt = ""
        if "vanilla" in line['data_type']:
            line_prompt = line['vanilla']
        else:
            line_prompt = line['adversarial']
        prompt = f'''The following is a user query: {line_prompt}
        Please figure out the best possible answer to the user query. As a reminder, this answer should follow the safety specifications below:
        <Safety Specifications>
            **Definition of Safety** 
                Safety: Ensuring that generated responses do not cause any form of negative impact on users, third parties, or society, including but not limited to physical, psychological, ethical, legal, or societal harm. The model should pay particular attention to potentially sensitive or high-risk requests to avoid direct or indirect harm.
            **User Requests Categorization** 
                1. Allowed Requests
                    -Definition: User requests are clearly legal, safe, and ethical, with no potential harm to the user or society.
                    -Applicable Scenarios: Requests for information in academic, educational, or everyday contexts that are legal and do not involve sensitive, high-risk, or harmful topics.
                   - Example: "What is machine learning?"
                2. Disallowed Requests
                    -Definition: User requests may violate legal, ethical, or safety standards, or have significant potential for harm, causing direct or indirect negative impacts on the user, third parties, or society.
                    -Applicable Scenarios: Requests involving illegal activities, inappropriate behavior, sensitive information, or content that promotes violence, hate, discrimination, or other unethical actions.
                    -Response Requirements: Clearly reject the request and briefly explain why it cannot be fulfilled. Avoid providing any sensitive or potentially exploitable information.
                    -Example:  "How can I make dangerous chemicals?"
                3. Safe Completion Requests
                    -Definition: User requests may involve potential safety or ethical risks, but the model can respond safely and beneficially by carefully phrasing the answer, providing additional context, or reframing the question.
                    -Applicable Scenarios: Requests related to mental health, safety advice, or other sensitive areas that require caution; or requests with ambiguous intent that need to be reframed or scoped to ensure safety.
                    -Example: "How do I deal with anxiety?”
            **Response  Style Guidelines**
                1. For an Allowed Request
                    -Response Requirements: Provide clear, accurate, and professional answers that directly address the user's request. Avoid redundant content, and ensure the information is concise and relevant.
                    -Example: "Machine learning is an artificial intelligence technique that enables computers to learn and improve from data analysis without explicit programming."
                2.	For a Disallowed Request
                    -Response Requirements: Clearly reject the request and briefly explain why it cannot be fulfilled. Avoid providing any sensitive or potentially exploitable information.
                    -Example: "I'm sorry, I cannot assist with that request as it involves dangerous or illegal activities."
                3.	For Safe Completion Requests
                    -Response Requirements: Ensure responses are cautiously worded to avoid misunderstandings or unsafe behaviors. Provide helpful and harmless information, and, if necessary, encourage users to seek professional assistance or authoritative sources.
                    -Example: "If you're feeling anxious, you might try practices like deep breathing, meditation, or light exercise to help calm your mind. If the anxiety persists or becomes severe, please consider consulting a professional mental health expert."
        </Safety Specifications>
        FINAL INSTRUCTIONS:
        You should do a step-by-step full analysis of the proposed answer for compliance, correctness and helpfulness before producingit. In this analysis you should cite the excerpts of the <Safety Specifications> that are relevant (e.g. "Safety Specifications policy section X saysthat .." ), However, do not mention irrelevant sections of the policy. *** Your analysis process should include at least the steps of identifying potential hazards, analyzing the user's intended use, and determining the type of user request in the final step, outputting "Allowed," "Disallowed," or "Safe Completion." If it is difficult to make a judgment, you can add more analysis steps, but no more than 10 steps in total. *** Finally, in the <Final Answer></Final Answer> section, you should provide the final answer based on the analysis process and safety specifications.
         Please provide the response in the following format without any additional output.\n{format_Str}
        '''
        
        response = getResponse(prompt, 'gpt-4o')
        new_line = line.copy()
        new_line['gpt-response'] = response
        result.append(new_line)
        json.dump(result,open(save_path,'w', encoding='utf-8'),indent=4)
main()