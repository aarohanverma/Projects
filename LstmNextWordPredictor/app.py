import streamlit as st
import torch
import sentencepiece as spm

# ---------------------- Model & SentencePiece Loading ----------------------
@st.cache_resource
def load_model():
    """Load the TorchScript model for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load("best_model_scripted.pt", map_location=device)
    model.to(device)
    return model, device

@st.cache_resource
def load_sp_model():
    """Load the SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load("spm.model")
    return sp

# ---------------------- Prediction Function ----------------------
def predict_next_words(model, sp, device, text, topk=3):
    if not text.strip():
        return []
    token_ids = sp.encode(text.strip(), out_type=int)
    if len(token_ids) == 0:
        return []
    input_seq = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_seq)
        probabilities = torch.softmax(logits, dim=-1)
        topk_result = torch.topk(probabilities, k=topk, dim=-1)
        top_indices = topk_result.indices.squeeze(0).tolist()
    predicted_pieces = [sp.id_to_piece(idx).lstrip("▁") for idx in top_indices]
    return predicted_pieces

# ---------------------- Streamlit App Layout ----------------------
def main():
    st.title("Real-Time Next Word Prediction")
    st.write(
        """
        Start typing your sentence below. When you finish a word (i.e. type a space at the end),
        the app will suggest three possible next words. Click on a suggestion to auto-complete your sentence.
        """
    )
    
    model, device = load_model()
    sp = load_sp_model()

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    
    user_input = st.text_input("Enter your sentence:", st.session_state.input_text, key="text_input")
    st.session_state.input_text = user_input

    if user_input.endswith(" "):
        predictions = predict_next_words(model, sp, device, user_input, topk=3)
        if predictions:
            st.markdown("### Predictions:")
            cols = st.columns(len(predictions))
            for i, word in enumerate(predictions):
                if cols[i].button(word):
                    st.session_state.input_text = user_input + word + " "
                    st.rerun()  # This triggers the refresh correctly
    else:
        st.write("Type a space at the end of your sentence to get next-word suggestions.")

if __name__ == "__main__":
    main()

