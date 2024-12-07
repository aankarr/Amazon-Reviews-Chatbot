import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def get_most_recent_model():
    latest_edited_file = max([f for f in os.scandir("finetuned/")], key=lambda x: x.stat().st_mtime).name
    return "finetuned/" + latest_edited_file

model = AutoModelForCausalLM.from_pretrained(get_most_recent_model())
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    "/scratch/bchk/aguha/models/llama3p2_1b_base", 
    padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

conversation = []
# Define the function that interacts with the agent
def interact_with_agent(user_input):
    global conversation
    example_inputs = tokenizer(
        [f"Question: {user_input}\nAnswer:"], 
        return_tensors="pt")
    example_outputs = model.generate(
        **example_inputs, 
        max_length=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id)
    response = '\n'.join(tokenizer.decode(example_outputs[0]).split("\n")[:2]).split("Answer:")[1].strip()

    conversation.append((user_input, response))
    # Format the conversation for the Chatbot component
    formatted_conversation = [(msg[0], msg[1]) for msg in conversation]
    return formatted_conversation, ""

# Define the function to clear the conversation
def clear_conversation():
    global conversation
    conversation = []
    return gr.update(value=[])

with gr.Blocks(title="Amazon Reviews Chatbot") as interface:
    gr.Markdown("# Amazon Reviews Chatbot")
    
    chatbot = gr.Chatbot()
    user_input = gr.Textbox()
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear")

    user_input.submit(interact_with_agent, inputs=user_input, outputs=[chatbot, user_input])
    submit_button.click(interact_with_agent, inputs=user_input, outputs=[chatbot, user_input])
    clear_button.click(clear_conversation, outputs=chatbot)

if __name__ == "__main__":
    interface.launch(share=True)