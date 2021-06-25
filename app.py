import streamlit as st
import pandas as pd
import numpy as np
import joblib
import emoji

pipe_lr = joblib.load(open("emotion_classifier.pkl","rb"))

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emojis = {"anger":emoji.emojize(":angry_face:"),
          "disgust":emoji.emojize(":disappointed_face:"),
          "fear":emoji.emojize(":face_screaming_in_fear:"),
          "joy":emoji.emojize(":grinning_face_with_big_eyes:"),
          "sadness":emoji.emojize((":pensive_face:")),
          "surprise":emoji.emojize((":star-struck:")),
         "neutral":emoji.emojize(":neutral_face:"),
          "shame":emoji.emojize((":worried_face:"))}

def main():
    st.title("Emotion Classifier")
    menu = ["Home","Monitor","About"]

    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1,col2 = st.beta_columns(2)
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emojis[prediction]

                st.write("{}:{}".format(prediction,emoji_icon))
            with col2:
                st.success("Prediction Probability")
                st.write(probability)


    elif choice == 'Monitor':
        st.subheader("Monitor App")

    else:
        st.subheader("About")











if __name__ == '__main__':
    main()
