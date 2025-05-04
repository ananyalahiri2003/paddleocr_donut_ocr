# PaddleOCR Donut OCR Streamlit app
Streamlit app to perform OCR using PaddleOCR or document understanding using Donut model.

The app extracts text from doc images. 
Produces outputs in Text and HTML format. 
_Note: As of 4May only single image upload and extraction is implemented._ 
## Setup

### Clone the repository
git clone git@github.com:ananyalahiri2003/paddleocr_donut_ocr.git

cd src

### Install dependencies 
`poetry lock --no-update`

`poetry install` 

All dependencies in pyproject.toml file. 

### Create .env 
Under src/ create `.env` file with 

HF_TOKEN={your_hf_token}


## Start the app
`poetry shell`

`streamlit run src/app.py`

This opens up streamlit interface on port 8501 and allows you to process. 

#### NOTE 
CPU is ok to use, things will be slow.  

## License
MIT License 

## Future Plans 
Add new models. 

Add support for multi-page pdf docs. 

_Pls add your wish list._ 

  
