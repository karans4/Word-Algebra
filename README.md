# Word Algebra
An edge compute word-vector based analogy calculator, running in the browser.

Basically, it takes word vectors, and illustrates how the difference between the two roughly corresponds with concepts.

EG : Let V(w) be a function which calculates a vector for the word w. V3 = V(woman) - V(man).  V3 + V(king) ≈ V(queen).

The "algebra" is done entirely in the browser, with a cache of around 8000 words loaded locally in the browser, 15 MB compressed.

The server exists to get the vectors of over 400,000+ less common words (trained 2024), but that's only used for inputs, not outputs.


![Beijing To China](beijing_to_china.png)

## How to Run
- Make sure you have python3, pip, and venv installed.
- Navigate in the terminal to the directory you downloaded this file in
- Run first_run.sh, This should set everything up for you, and if it works, then skip to step 8.
- Create a venv with `python3 -m venv venv`
- If you are not in the venv, make sure you do source `./venv/in/activate`
- Install the requirements with `pip install -r requirements.txt`, also install `gunicorn` on production
- Run `python setup.py` to initialize the vectors (may take 20 to 40 min to run)
- You can either run `flask --app app.py run` when testing, or `gunicorn --workers 4 app:app --preload --bind 0.0.0.0:5000` on production servers.
	* If you do preload, you might want to set embeddings.npy to read only with `chmod 444 embeddings.npy` to save memory.
	
	
## Tips
* You may want to use cloudflare or a CDN for caching. It brings the bandwidth and load on the server to a fraction of a percent what it would be otherwise.

