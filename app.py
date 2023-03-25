from sentence_transformers import SentenceTransformer
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import torch
import clip
import os


# # Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_model():
  model, _ = clip.load("ViT-B/32", device = device)

  return model

def get_preprocess():
  _, preprocess = clip.load("ViT-B/32", device = device)

  return preprocess
model, preprocess = get_model(), get_preprocess()
nlp_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

image_features = np.load('./image_features.npy')
# Encode the text description

messages = pd.read_json('news_with_photo_index.json')
encoded_messages = np.load('messages_encoded.npy')
sentences = pd.read_csv("sentences.csv")

st.set_page_config(page_title= 'ChatSearchAI', layout = 'wide')
# tab1, tab2 = st.tabs(["Search from images", "Further redevelopment"])
st.title('AI powered Chat Search - :red[ChatSearchAI]')
col1, col2, col3 = st.columns((5,1,5))
with col1:
  st.title('Image Search')
  st.info('Note that model\'s performance is dependent on OpenAI\'s CLIP model', icon="ℹ️")
  st.info('Since application loads OpenAI\'s CLIP model, application might work slowly', icon="ℹ️")
  # with tab1:
  search_query = st.text_input('Type what are you looking for in images:')
  num_of_images = st.number_input('State how many images you want to see', 
                                  min_value=1, max_value=10, value=3, step=1)
  button = st.button('Search and show results')
  if button:
      text = search_query
      text_input = clip.tokenize(text).to(device)  
      with torch.no_grad():
          text_features = model.encode_text(text_input)  

      # Compute the cosine similarity between the text and images
      similarity = torch.nn.functional.cosine_similarity(torch.from_numpy(np.array(text_features)), 
                                                      torch.from_numpy(image_features), dim=-1)
      indices = similarity.argsort(descending=True)

      image_paths = list(map(lambda x: './images/' + x, 
                          os.listdir('./images')))
      # Print the most relevant images
      # c = st.container()
      # with c: 
      for i in range(num_of_images):
          index = indices[i].item()
          image_path = image_paths[index]
          # print(f"Rank {i+1}: {image_path}")
              # Image.open(image_path)
          img = Image.open(image_path)
          h, w = img.size
          new_height  = 300
          new_width = new_height * h / w
          img = img.resize((round(new_width),new_height)) #w,h
          st.image(img)
          message_id = messages[messages['jpg_name'] == image_path[9:]]['message_id'].values[0]
          st.write(f"[see message](https://t.me/bloomberg/{message_id})")
      # with open(image_path, 'rb') as file:
      #     img = Image.open(file)
      #     img.show()
with col3:
   st.title('NLP Search')
   nlp_search_query = st.text_input('Type what are you looking for in messages:')
   num_of_messages = st.number_input('State how many messages you want to see', 
                                  min_value=1, max_value=15, value=5, step=1)
   nlp_button = st.button('Search and get results')
   if nlp_button:
    nlp_similarity = torch.nn.functional.cosine_similarity(torch.from_numpy(np.array(encoded_messages)), 
                                                            torch.from_numpy(nlp_model.encode(nlp_search_query)), dim=-1)
    message_indices = np.array(nlp_similarity.argsort(descending=True))[:num_of_messages].tolist()
    #related_messages = sentences.iloc[message_indices]
    for x in message_indices:
        st.write('" ' + sentences.iloc[x].sentences + ' "')
        st.write(f"[see message](https://t.me/bloomberg/{sentences.iloc[x].id})")
with st.expander("Further possible development perspectives"):
   st.write("""
   The application can be further redeveloped using 
   1) NLP methods such as "Question - Answering" models.
   2) Searching inside video and audio data
   3) Question answering based on images, etc.
   
   Open to collaboration and pull requests. See [GitHub repo](https://github.com/huseyn-guliyev/ChatSearchAI)
   """)
