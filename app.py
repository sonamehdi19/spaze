import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
import string
from string import digits
import re
from sklearn.utils import shuffle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense,Embedding, Concatenate, TimeDistributed
from tensorflow.keras.models import Model,load_model, model_from_json
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import pickle as pkl
import numpy as np
import os, urllib

from PIL import Image
from model.attention import AttentionLayer

json_file = open("model/spelling_model_l.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_loaded = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})
# load weights into new model
model_loaded.load_weights("model/spell_model_weight_l.h5")


Eindex2word, Mindex2word = pkl.load( open( "model/spell_word_index_l.pk", "rb" ) )

inputTokenizer, outputTokenizer = pkl.load( open( "model/spell_tokenizers_l.pk", "rb" ) )
Mword2index = outputTokenizer.word_index
Eword2index = inputTokenizer.word_index

latent_dim=500
# encoder inference
encoder_inputs = model_loaded.input[0]  #loading encoder_inputs
encoder_outputs, state_h, state_c = model_loaded.layers[6].output #loading encoder_outputs
#print(encoder_outputs.shape)
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# decoder inference
# Below tensors will hold the states of the previous time step
#decoder_inputs_t = model_loaded.input[1]  # input_2
#decoder_inputs = tensorflow.identity(decoder_inputs_t)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(57,latent_dim))
# Get the embeddings of the decoder sequence
#decoder_inputs = model_loaded.layers[3].output
decoder_inputs = model_loaded.layers[3].output  # input_2

for layer in model_loaded.layers:
    layer._name = layer._name + str("_2")


#print(decoder_inputs.shape)
dec_emb_layer = model_loaded.layers[5]
dec_emb2= dec_emb_layer(decoder_inputs)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_lstm = model_loaded.layers[7]
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
#attention inference
attn_layer = model_loaded.layers[8]
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
concate = model_loaded.layers[9]
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_dense = model_loaded.layers[10]
decoder_outputs2 = decoder_dense(decoder_inf_concat)
# Final decoder model
decoder_states_inputs = [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c]
decoder_states=[state_h2, state_c2]


decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],[decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = Mword2index['<']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
          break
        else:
          sampled_token = Mindex2word[sampled_token_index]

          if(sampled_token!='>'):
              decoded_sentence += ''+sampled_token

              # Exit condition: either hit max length or find stop word.
              if (sampled_token == '>' or len(decoded_sentence.split()) >= (26-1)):
                  stop_condition = True

          # Update the target sequence (of length 1).
          target_seq = np.zeros((1,1))
          target_seq[0, 0] = sampled_token_index

          # Update internal states
          e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=Mword2index['<']) and i!=Mword2index['>']):
        newString=newString+Mindex2word[i]+''
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+Eindex2word[i]+''
    return newString

# function for preprocessing sentences
def preprocess_sentence(w):
  #w = unicode_to_ascii(w.lower().strip())
  w=str(w)
  w = w.lower().strip()
  w = re.sub(r"([?.!,??])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,??]+y??ukenq??hzxjf??vaproldcg????smit??b??Y??UKENQ??HZXJFIVAPROLDCG????SM??T??B??", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  #w = '<start> ' + w + ' <end>'
  w = '< ' + w + ' >'
  return w

def spell_check(w):
  w= preprocess_sentence(w)
  X_test_word = inputTokenizer.texts_to_sequences([w])
  X_test_word = pad_sequences(X_test_word, maxlen=57, padding='post')
  w= decode_sequence(X_test_word[0].reshape(1,57))
  w=w.replace(" .","").replace("<","").replace(">","").strip()
  return w


icon = Image.open("static/logo.png")
st.set_page_config(page_title='Az??rbaycanca Orfoqrafiya Yoxlama Platformas??', layout='wide',  page_icon=icon)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

    .menu .nav-item .nav-link.active {
        background-color: #ffd04b;
    }
    .nav-link.active .nav-link-horizontal
    {
    font-family: 'Poppins', sans-serif;
    }
    p{
    font-family: 'Poppins',sans-serif;
    line-height: 1.929;
    font-size: 18px;
    margin-bottom: 0;
    color: #888;}
    h2 {
    font-family: 'Poppins',sans-serif;
    font-weight: 600;
    color: rgb(49, 51, 63);
    letter-spacing: -0.005em;
    padding: 1rem 0px;
    margin: 0px;
    line-height: 1.4;}

    .css-rncmk8 {
    display: flex;
    align-items:center;
    flex-wrap: wrap;
    -webkit-box-flex: 1;
    flex-grow: 1;
    --fgp-gap: var(--fgp-gap-container);
    margin-top: 50px;
    margin-right: var(--fgp-gap);
    --fgp-gap-container: calc(var(--fgp-gap-parent,0px) - 1rem) !important;
    }

  </style>

    """,
    unsafe_allow_html=True
)


footer="""<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}



a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}


.footer {
position:fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
z-index:2;

}
</style>
 <div class="footer">
                    <p class="footer-text m-0">
                      ?? 2022 | Developed by 
                      <a href="https://github.com/sonamehdi19" target="_blank"
                        >sonamehdi19</a
                      >
                    </p>
  </div>
"""
st.markdown(footer,unsafe_allow_html=True)




selected=option_menu(menu_title=None, 
    options=["Ana s??hif??", "Aplikasiya", "Haqq??nda"], 
    icons=["house", "spellcheck", "info-circle"], 
    menu_icon="cast", 
    default_index=0,
    orientation="horizontal"
)

if selected== "Ana s??hif??":
  st.image(Image.open('static/main.png'),use_column_width=True)
  st.header("Niy?? SpAze?")
  c1, mid, c2, mid2 = st.columns([5,4,20, 3])
  with c1:
    st.image('static/team.png', width=300)
  with c2:
    st.markdown("Orfoqrafik s??hvl??rin ??ks??riyy??ti sosial mediada v?? e-po??tlarda ??t??r??l??n m??lumatlarda olur. Axtar???? sisteml??rind?? v?? t??rc??m?? sisteml??rind?? orfoqrafiyan??n korreksiyas?? da vacibdir. Kontekst t????kil ed??n c??ml??l??rin ifad??l??ri v?? m??nas?? ??ox vacibdir. ??oxlu qrammatik v?? orfoqrafik s??hvl??ri olan m??zmunu oxumaq he?? d?? xo?? deyil.Proyektin ??sas m??qs??di orfoqrafiya yoxlay??c?? alqoritm haz??rlamaqd??r ki, ??z bloqlar??nda, sosial ????b??k??l??rind?? Az??rbaycan dilind??n istifad?? ed??n istifad????il??rin yazd??qlar?? m??tnl??ri t??z??d??n vaxt s??rf edib g??zl?? yoxlamadan orfoqrafiyas??na ??min olsunlar.", unsafe_allow_html=True)

  c1, mid, c2, mid2 = st.columns([20,1,10, 2])
  with c1:
    st.markdown("**SpAze** platformas?? Az??rbaycan dilind??n d??zg??n istifad??nin formala??d??r??lmas?? ??????nd??r. Komp??ter v?? telefonlar??m??zda Az??rbaycan klaviaturas??n??n m??vcud olmama???? v?? iki ??lifba aras??nda daim d??yi??iklik ed??rk??n yaranan ??a??q??nl??q v?? ya orfoqrafiya qaylar??n?? d??rind??n bilm??m??k yaz??da s??hvl??r etm??miz?? g??tirib ????xar??r. Bu is?? dilimizin zamanla modifikasiyalara u??rama????na s??b??b ola bil??r. ", unsafe_allow_html=True)
  with c2:
    st.image('static/dayflow.png', width=350)


elif selected=="Haqq??nda":
  st.image(Image.open('static/main.png'),use_column_width=True)
  st.markdown("""T??klif olunan model, h??r bir s??hv, d??zg??n c??ml?? c??t??n??n simvol i??ar??sin?? ??evrildiyi v?? model?? qidaland?????? ard??c??ll??q modelin?? xarakter ??sasl?? ard??c??ll??qd??r. Model 3 hiss??d??n ibar??tdir - Kodlay??c??, Dekoder v?? Diqq??t mexanizmi. T??klif olunan modeld?? h??m kodlay??c??, h??m d?? dekoder modeli embedding v?? LSTM qatlar??ndan ibar??tdir. Diqq??t mexanizmi m??lumat?? Kodlay??c??dan Dekoder?? da????yan kontekst vektoru yarad??r.
T??klif olunan model c??mi 3000 real d??nyada s??hv yaz??lm???? s??zl??r ??z??rind?? t??sdiql??nir v?? d??qiqlik proqnozla??d??r??lan s??zl??rl?? real s??zl??r aras??nda Levenshtein redakt?? m??saf??sinin yoxlan??lmas?? il?? ??l????l??r. M??saf?? 0 o dem??kdir ki, model s??z?? ??ox d??zg??n proqnozla??d??r??b. M??saf??nin 0-dan b??y??k olmas?? o dem??kdir ki, model s??zl??ri m????yy??n d??r??c??d?? d??zg??n, lakin m????yy??n redakt?? m??saf??si il?? proqnozla??d??r??b.
Bu t??dqiqatda Az??rbaycan dilinin orfoqrafiyas??n??n korreksiyas??na diqq??t mexanizmli kodlay??c??-dekoder modeli t??tbiq edilmi??dir. ??yr??nm?? v?? test datalar?? s??hv v?? d??zg??n c??ml?? c??tl??rinin ard??c??ll??????ndan ibar??tdir. Model real s??z m??lumatlar?? ??z??rind?? s??naqdan ke??irilir v?? ??mumi n??tic??l??r m??saf?? 0 ??????n 75%, m??saf?? 1 ??????n 90% v?? m??saf?? 2 ??????n 96% t????kil edir.""", unsafe_allow_html=True)


else:
  c1, mid, c2 = st.columns([1,1,20])
  with c1:
    st.image('static/logo.png', width=185)
  with c2:
    st.markdown('<h1 style="text-align: center;">SpAze - Az??rbaycanca Orfoqrafiya Yoxlan???? Platformas??</h1>', unsafe_allow_html=True)
  col1, col2=st.columns(2);
  text = col1.text_area(label ='??lkin m??tn:',placeholder="M??tni bura daxil edin...", value='', height=185, max_chars=None, key=None)
  
  if st.button('Yoxla'):
      if text == '':
          st.warning('Z??hm??t olmasa m??tni daxil edin.') 
      else: 
          result=spell_check(text)
          col2.text_area(label ='D??z??ldilmi?? m??tn:',value=result, height=185, max_chars=None, key=None)
  else:pass

