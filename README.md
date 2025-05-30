# Dimpon_hse

## Project Description

The purpose of this work was to develop a classification model that allows predicting the mental health condition of employees at remote work.For model interaction, a user-friendly web interface was developed using the Streamlit framework. Additionally, a FastAPI-based RESTful service was implemented to facilitate future integration of the model into external systems.

## Streamlit app

The application is available at the link below
https://dimponhse-appyb9pn7fx8rlp5zpwbztr.streamlit.app/

## Repository Structure
```text
Dimpon_hse/
├── fastapi_app/ # Backend: FastAPI API for interacting with the ML model
│ ├── api.py # Main API code
│ └── test_api/ # Tests for the API
│
├── streamlit_app/ # Frontend: Streamlit app for visualization
│ └── streamlit_app_app.py # Main Streamlit application script
│ └── requirements.txt # Python dependencies list for Streamlit app
│
├── model/ # Trained ML model 
│ └── model.pkl # Serialized ML model (pickle format)
│
├── notebooks/ # Jupyter notebooks with a pipeline of model classification development
│ └── final_model.pkl # Final model file with a full pipeline of model development
│
├── .gitignore # Git ignore rules
└── README.md # Current project documentation file

