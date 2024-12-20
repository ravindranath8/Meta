import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Title and description
st.title("meta-llama/Llama-3.2-1B Demo")
st.markdown("Interact with the Meta-Llama/Llama-3.2-1B model directly from your browser!")

# Hugging Face Token Input
st.sidebar.title("Authentication")
hf_token = st.sidebar.text_input(
    "Enter your Hugging Face Token:", type="password"
)

# Load model and tokenizer
@st.cache_resource
def load_model(token):
    if not token:
        st.error("Please provide a valid Hugging Face token!")
        st.stop()
    
    model_name = "meta-llama/Llama-3.2-1B"  # Replace with the actual model identifier
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    
    # We use 'torch.float32' to avoid potential issues with float16 compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        use_auth_token=token,
        torch_dtype=torch.float32,  # Changed to float32 for better compatibility
        device_map="auto"           # Automatically map model to available devices
    )
    return model, tokenizer

if hf_token:
    model, tokenizer = load_model(hf_token)

# User input
prompt = st.text_area("Enter your prompt here:", height=150)
if st.button("Generate Response"):
    if prompt.strip():
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Check for GPU/CPU device availability and move the model accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200, num_return_sequences=1, do_sample=True)
        
        # Decode and display the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("Model Response:")
        st.write(response)
    else:
        st.warning("Please enter a valid prompt!")

# Footer
st.markdown("Powered by [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers).")
