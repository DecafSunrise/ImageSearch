# ImageSearch

I found [this neat article](https://python.plainenglish.io/creating-an-image-search-app-in-python-using-clip-and-streamlit-854933d742ca) the other day, and figured it could be a good way to wade through a collection of ~thousands of pictures I've saved over the last decade. 

# Setup:
1. Follow the instructions in the [linked article](https://python.plainenglish.io/creating-an-image-search-app-in-python-using-clip-and-streamlit-854933d742ca). I had to downgrade Pillow (image manipulation library) to 9.0.0 because of weird instability.
2. Ensure you've got DotEnv set up (`pip install python-dotenv`), and an appropriate .env file in repo directory, pointed at the working directory of images.
3. Run `python get_embeddings.py` to build an embedding representation of images we can use for search and stuff. If you add or remove images, you'll need to run this again. It ran fairly quickly (~2-3 minutes for ~1500 pictures) in tests. It doesn't know what to do with .avif files; consider [converting them to PNGs!](https://github.com/DecafSunrise/Convert-to-PNG).
4. `streamlit run app.py`
5. Enjoy!
