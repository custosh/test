# Prime-pair Hyperbola Explorer

A Streamlit app that plots products of same-bit-length prime pairs against the ideal hyperbola `p*q = N`.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed in the terminal (usually http://localhost:8501).

## Publish it online with Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the app file to `app.py`.
5. Select **Deploy**. Streamlit will give the app a `*.streamlit.app` URL.

The deployed app runs on Streamlit's server. Keep in mind that larger bit counts and large prime pools can use substantial compute and memory; the app's sampling controls help limit plotting work.
