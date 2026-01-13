import gradio as gr
from src.complaintRagChain import ComplaintRAGChain
from src.complaintVectorStore import vector_db
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# 1. SETUP THE LOCAL LLM (Same logic as Task 3 for consistency)
print("=== Initializing Backend for UI ===")
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager" # Stability fix for Phi-3
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# 2. INITIALIZE THE RAG ANALYST
# vector_db is imported from your existing script
analyst = ComplaintRAGChain(llm=llm, vector_db=vector_db)

# 3. DEFINE UI FUNCTION
def process_query(message, history):
    """
    Main function to handle user questions.
    Returns: The AI response and the formatted source documents.
    """
    # Run the RAG pipeline
    response = analyst.query(message)
    
    answer = response['result']
    sources = response['source_documents']
    
    # Format sources for the UI
    formatted_sources = "### 📚 Evidence / Source Documents:\n"
    for i, doc in enumerate(sources):
        complaint_id = doc.metadata.get('complaint_id', 'Unknown')
        product = doc.metadata.get('product_category', 'General')
        content = doc.page_content[:300] + "..." # Truncate for readability
        
        formatted_sources += f"**[{i+1}] ID: {complaint_id} | Category: {product}**\n"
        formatted_sources += f"> {content}\n\n"
        
    return answer, formatted_sources

# 4. BUILD THE GRADIO INTERFACE
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 CrediTrust: Intelligent Complaint Analyst")
    gr.Markdown("Transforming customer feedback into actionable insights for Product Managers.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot_output = gr.Textbox(label="Asha's AI Assistant Response", lines=10, interactive=False)
            user_input = gr.Textbox(label="Ask a question about customer complaints...", placeholder="e.g., Why are people unhappy with Savings Accounts?")
            with gr.Row():
                submit_btn = gr.Button("Analyze Complaints", variant="primary")
                clear_btn = gr.ClearButton()
        
        with gr.Column(scale=1):
            source_display = gr.Markdown("### 📚 Evidence\nSource documents will appear here after analysis.")

    # Define Button Actions
    submit_btn.click(
        fn=process_query, 
        inputs=[user_input], 
        outputs=[chatbot_output, source_display]
    )
    
    # Reset everything when clear is clicked
    clear_btn.add([user_input, chatbot_output, source_display])

# 5. LAUNCH THE APP
if __name__ == "__main__":
    demo.launch(share=True) # Set share=True to get a public link for facilitators