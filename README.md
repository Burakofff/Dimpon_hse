# Dimpon_hse

## Repository Structure
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

