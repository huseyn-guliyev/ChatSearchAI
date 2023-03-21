import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
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


image_features = np.load('./image_features.npy')
# Encode the text description

messages = pd.read_json('news_with_photo_index.json')

st.set_page_config(page_title= 'Reverse Image search - ChatAI')
# tab1, tab2 = st.tabs(["Search from images", "Further redevelopment"])
st.title('AI powered Chat Search - ChatSearchAI')
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
    for i in range(num_of_images):
        index = indices[i].item()
        image_path = image_paths[index]
        # print(f"Rank {i+1}: {image_path}")
            # Image.open(image_path)
        st.image(Image.open(image_path))
        message_id = messages[messages['jpg_name'] == image_path[9:]]['message_id'].values[0]
        st.write(f"[see message](https://t.me/bloomberg/{message_id})")
    # with open(image_path, 'rb') as file:
    #     img = Image.open(file)
    #     img.show()
with st.expander("Further possible development perspectives"):
   st.write("""
   The application can be further redeveloped using 
   1) NLP methods such as "Question - Answering" models.
   2) Searching inside video and audio data
   3) Question answering based on images, etc.
   
   Open to collaboration and pull requests. See [GitHub repo](https://github.com)
   """)