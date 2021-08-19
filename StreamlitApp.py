
print('the begining')
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
# import modeling libraries.
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from nltk.tokenize import sent_tokenize
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.util import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.util import ngrams
# Construction from scratch
from spacy.vocab import Vocab
from spacy.language import Language
nlp = Language(Vocab())
from spacy.lang.en import English
nlp = English()
import spacy
from nltk.stem import WordNetLemmatizer

#******************************************************************************************************

#Preprocessing function
# convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub("@\S+", "", text)
    re.sub("\$", "", text)
    text = re.sub("https", "", text)
    text = re.sub("co", "", text)
    text = re.sub("https?:\/\/.*[\r\n]*", "", text)
    #text = re.compile(r'https?://\S+|www\.\S+')
    #text = re.sub("#", "", text)

    return text


# tokenizer, pos tagging and entity recognition


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

# ***************************************************************************************

#Create the horizontal sections of our app
header = st.container()
overview = st.container()
datadesc = st.container()
plot = st.container()
model_prediction = st.container()
endnote = st.container()

#The first section
with header:
    st.title('ClimateTalks')
    st.write('Supervised machine learning models to classify climate change related text data.\n\n\n')
    col1, col2 = st.columns(2)
    with col1:
        url  = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxQUExYUFBQXFxYYGRsbGhkZGRocHxsfHhkYGRkbHhgbHyoiGhsnHBgZIzMjJystMTAwGSE2OzYvOiovMC0BCwsLDw4PHBERGy8oIigvLy8vODEvLzExMi8yLy8vLzgvLy84MS8vLy8vLy8vLy8vLy8vLy8vLy8xLy8vLy8vL//AABEIAMcA/QMBIgACEQEDEQH/xAAbAAACAgMBAAAAAAAAAAAAAAAEBQMGAAECB//EAD4QAAIBAgUCBAUBBgUDBAMAAAECEQMhAAQSMUEFURMiYXEGMoGRoUIUI1KxwfAVYtHh8QdDgjNyorIWksL/xAAaAQACAwEBAAAAAAAAAAAAAAACAwABBAUG/8QAMBEAAgICAgECAwcEAwEAAAAAAAECEQMhEjFBBFETFGEiMnGBkaGxBcHh8CNC0RX/2gAMAwEAAhEDEQA/APH1xKoGIApx2inGdo2RYSFx2FxxTBxMq4U2OiaAx2Fx2KeJFTC3IOiNUx2tPE9NcF08vNxgXIuhf4eJaSYYDLzxiajkZwPIlAtKlBnBtTI8jnBdLJ22wzymVlY5GCixUkVtsniF8pi2fsE8YgqdNw6ImRUnyRxBUypxaquQjEX7BPGNWMzTKoMqZxxWpRi3p0onCDqeWKsQca8ezPJirRjXhYIRMFJQ5xqjAS5ixqOOSuGNSniMUx2w+OMW5gJXGCmcPui9HfM1BSW3Mxxi5t8FGlS3Gs/Mf9sSThB02RcpK0jzXKwrAkavQ7fbnHoVDorZmlTpgBBYsYO3Me+FeR+EHaqLhksS23vvfHpfTkp0kgX9TvhWfIopcewoR5St9EPTug0MsoCIJHJ3xmbzcWGMzfUAbTiBEm+MLbe5GjS0hecu1Q3sMdVaCgQBg91xBUXFciuIorU8CtSw1qJgVqR7YNSAcTyOlXXE6OMJw2J/EI5xgljOtHINNQGCEYYUNmCwjnGqWYmx42wt4m0M+IrH6LONlYwl/bCWBJsMTHqNtj64U8Mg1kiPECgSTg7KaeDPp/tiqmqGAvcYO6VWiZYhuPXAyx0rCU03RdKOQkSMFUchOK50/OFKgIMA7yd8X3p9VaiSsCNzbCWGCUMh3wyy2Rg4Oy2XnthnlcrNrYZARkkKx08YW5+mFJ8kkeoxdVylsR1OnoblZ+mNSiZXIoVZJBIQx/fAxvL5B7EiJ4xfj05QhhYnANLKL7xh0NCZ7K6vTzpNsUjquRZnJN8exvlwKZPpivjpGskaZxrxyrbMs0+jzFOmHtiatktIx6dT+FiLkKPc7YWdR+Gg5AFQGdgB+ZPA+uNcM0REoSXg8wrU744Wn6Y9k6d8DUqYkr4jfxECB6hTOIaPwrllaPD1P34/Gxvhi9Zj9mT5efkrfwnnqXlRQqNaSBcntO5OLV1ir4dI7FiLajGF+a+DFHnpSlSbXGkHvH9cQ534VdobMZk1COAIAwmbxzly5f8AocVOK40LslmkT/u+S5YDufXtbCjqfxcwMU1ECZnfD1vhHLSCNQFpEm8fXnHOY+HsuqkMu83EgxwJ7AYNSxXbtg8Z1RTun57MVnJWSZ3GwnFpo18zSAU6XYiwF/ue2OqHS7fuk0oYuJuPrh70/pSUlkbxuTJwOXJD2LhCVizLVMw16iBY/SGmfxhnSa1xHpjgVDzfBNMgjtjLJp+B8VXkDeqO18RxPIx3m4vhVVrwcUiPR4iDjoHEeOgcINaZIoOOxjhXx0rYFhonKYJy9GRpIg+uBUcxGCVBb3wqVj4UE0KYuGH1w1y2VBUEETxPHpfjCzLdjhvQRjsJH8vbGeabNEGkF5PJawQR5ptHOLH8OBqRMCTIt2ve2EtBiGnQOJG38jhp03q5p+VkBP8AHeR9Np9YwhwmG3Gi9UaNtQWMMumhyZny+uK/kOpu0HdeRbFqyFGQCNsaMeN+TLkaGentjSUzgrLU8T+FjUomRsWtlyVP1wvp5I4sgpYhp0MGlQD2Ks5lYQDHGUyZSPX+5OHdWjJHpjaUQMMvQvjuwHMU7QPzhHlukuH1a1EdhM8wdsWd6c4jbL3nnAxk10FKClTZC22BM05A2wdVYKML3cPvb0xSRbYgrZ1w+1sSVKivub9owbmMmO04hTpk3P2w1NCqYqKQdKgnElTIyPPhyNKiFGFudk4LkycUiFYiBsMAZurGJi2n9QGA62YXfUDi0tguWiBapONmsF3wpzOeGsKNzglco53IjBuFdi1O+jrMVi22FtbLvOGioqizXwtznVNLQxg4kYt9EbVbPHxROOhQ9RgrwjAJFtsdfszAx5fuMYOZ11h+gKKGOloYK8H12P8AcYMyuUd28iz9Bz/XASyUg44bdUL6VAd4xKigHfD/ACvw7WLaUXURvp/IB2OGub+BKqjUkVBztK+hE2xnl6nGnTkPXp2l0VCnUM3Jw26bm4IF4wdR+FH0sXYKwiBvPfGZTocHzMfoMC/UQ9wl6ebH2QGq2H+T6IG9sJemikhF3DDeVBn3GLl0PqSgi9vbALOnok8Eo7Juk9G0+bYcDufQYs/S6Ee3btibIVKbwZvxPGGlOmBsMa8VvdmDLPw0ZSSMSxisfHvU3o5f92wQuSNd5UBSxiObfacAf9NetPWpsru1TSEcO5BeHmzEGD8u4tvFoh/JXQni2uRca1dUEsQPc4jymYDqCPqJBgwDEix3xXutdVRawpss+YXuP+2z7gX22wi6112pT8MUGKBi+rVqNwVAiRJi47DGd+oanTWg1itF6/b18Q09LWMaraZ0ho3nYjjnBhxQOk9aOjxKrEF6mlvVilMbe5jFp6j1RA4oq8VNyIMxpLSDEcDD4ztNgONOhrGBc3mVpqWYgKoJJOwAuT9sKMjnazPJdSmtljTeAxXeY47DE/xY+jK5h4mKT2/8SMFCcZdC5pxIaPUqNQapmbrNiwiQQszEd7+mE3Vq2pgKYIEiYs3aJ+uJumV0qVPBUEadQk86DoJt6gnbHXUsqV4P0xp9K1Lb7MfrHKOl0M6RRFgm/qZP3OFGY69SBYMwGnFV+Jc1Uo02cqTBWxJE6mC9vXCzO5KophwL8qxP9BjRHDiU1GctvoR8xmlByhHUeyw534qog+WT6jAdXq+uNLBQRYtis1Mv2H97YFq0SOD9sbPlMPSezP8AN5Wra0M63UnLaDBYmBBMffFfr9RfWyncEg9sT5HMAZmnSi8zPaxYD7DAdd4zDxvqP84/rhUpRjkUY1Vb/U0QhKWNyld+P0JVzThgy2Pc3/nh3ketVW8lRlv+q2E2sRsZ97fbG62b1AArb0t+cap4lLVGaGVrplj/AGjL05DEMe+Ba/WMsTcA+6zhItSnyPzt+L4Ny1KlHzoffGaWBR27NCyt9UJ8l0tahACgE9pgfQ2OG2X+GlLQWCHi6j8DEecKLtU83YL/AFnAOS6rUavoLNoUeinUdpMlgInbtjxsVly/ddfie5yyx41b/Asq/DWWQhmkwLlTI+w5+2GWWoZXWCtFdoa0N2O9jitv8Q5jwwzuNA0rcgmSrFfKoEyFa/ocS1M4WpUqiZhGNSoFVaZZXMSG8sBtIMehkXwl+nzN1Jv2AeXHGN2rLvS6jQpNBQKVjYRuJFubRhF1jq7tmvDSDTlVAUG5IJN+GBG3qfpX87SqBTV8xJUeYm87btJ/sYUUcy9OqWBKEVAymwuD5YOwvyZw/H6KK3duv3McvUPTRYeqo9NgG1XvwIEhZknuQPrhtlejvtq/livfEOfr1kp1KjguKRAhYJJqU2Vuz2TcgXJtex/7fUcMsqsDUbrT0mZnUTM6o9bcYasDUV19Q/mZNvQwqIquVNGs8GCVCBbbwWcExfYcYhfLUjS2ElJuQSfLOx2+nftGFWSzFdWUMCaZUWNwZL6iWNxf1G2GmTyr1YQ6jTUMtoNwihbbbsv6pgNe2LarpinKU9s9M/ZRl6T1W1FUUsfYCTih/E3xLVzEMjmktNSVCO0zEltYjzWAFrXvc4r/AFnrGcrVnjM1woQKEDkK3lMyiwpmfXcdhjjpdVnpGmA+vw2F1ETeQDMyZ5vvgq4K0york/t9lx671g5nJIlTUayNUZrL8pWroiLHyFJtzhV8FfFVPIpWqshqKtKmG0RJIqimIJgR+9P2+4HU+rKiMNmYLCje66W9rT9vphZ8L0RQUq/7zWRrUKRckDSJIBghWuMGptfbZTxxrgj1T9sXMNTrhSoqEHSwBKjQy3ImJIFx39cI/i3SHpKrgNDSCDMFlAMR3X8YL6LnzSo0VZhpWnEQ0yumfte4/pir/wDUDMlqg0sJFMX7anIvzbGdNzmHUYrroKq1NNGisi2aIkWEK1EMb7AKJxZsz1OnV6kr0mDoaR8w2kBu+/vtimVck1KjT1DVWqKW8pJJgxp0zYkgfcc466RV0fs9QkrDhCl9tLDefMP9AcdGDi8XZhnGXxD0fJZsKrM2yvUY7bB2J/Ax598S/FoJr0qdV28aoe8aDU1CdXBSAI2mLYd5/OhcvVbaUfTYC51/aLGPXFSyjK+QqFgrVVIaIUkQQNQ7HRO3E4rBFR2/dEy70vYvGU6nTy1Zq1UxTD1ASONdcoD7S2EC9TquXcZiqQatUr52I0itUCeVpAGmLRiBswmYV1qfKxJImP8AvVKg9Z2Njxhbk8rRXN5agrE065YOGYMLcibET3nE4tqk2ndkTinbVqqL/wDG9WinTfEdFZyuXN4Bc66TGPXc4ESpoIbeabKJv5tJfVJuLIfuMVz44ytGpTp0ssfN4/gsoJCWGoeTaJXcdsF5vNs7NTpOi1aCtUdXBugpsrqInzw4Iwcm3KL/AB/QCEUoyS8lSz2cZKtWm7eJKINbRNmR79+Riw/D2UPg1AwdVqwAw0sBEj5VbV+MVX4g6dWplK9TRpqxBVpg2MEQINibWsb4u3QDNKl/fJw3PkpRlHe1+qAwwu4v6/uU7KEft6SRaRPtSj+eBwurNle9SPuyjAmWq6s1YExUb1/XP8hhl0/IVamZNRFlPFjVIgQ6zzNpnbGi6nbfj+4lxvHS9/7Ds9EQq373S4JGki3tP9cJa+WK7x9Di3Vi6gjVRgcljO/cjEFHpIclqisw7Jt9+cdOGfjbkzkvDdKKKeyYhKYtWe6QCx0IFUd2/wCYOIB0Wny9/S/5w35nHVlLFNOqCspWoU6lMtTRUViWUEElYQm5APv9cVvq3UKdXPV61JCtJmXQDHCqrCATyCR74U5/NB9LVSdjGkaBv/mkn9Nx67b4LoVFrVEAA1hLhacC0ywA0jUdSkn0NseSjjUd/Sj1byc3d/5IOq5SoWWqFOhgsCT+kKPtJK6pmdXe+dDyjgk+QaCKtxcxEC99tVu8YcdNFRFY+EzMIKO6jTMGY1qRuAQVsCTM4g0uFdhp1FhBLeUAmSCF2mfewtc4LlSoH7zuifqmZrVEVNJglCpFlgOCLCBqgNYkni2IAWpsHY0oVlZgzqbDzad/MbrYXm2+IaiuUXUV8W3meotvQAmNp2GNGofFWp4qAqdesCQSY321G143k3GBtVVlxtPoZdSy9KpXmnUfw2GkBUqkqAq7HQUIsTv3meRc0mWARFqVKwQtLMoUKSTYAMGM77mMZmKxqMxauzc2VrzbyobRPMRffsG2RuQKt2uFCiTbcgEx+cApJabZcm27SLX1Hr1I06BVCyqFAY6VIgLG29jAP2MknD2fGA/eRosJYKNwfMkgTfcifeMedUelVGURVrERssxFjsSoAkKft2xOfhKQC3iEG8kqAB/5Xme2EtY15oL4kvMf3LTnaAJilVy402ZmrKongAPpkETJvH4CnO5qqF8tfJpDEELmabFpOkNopjzQDMntMcYB/wDwmoBZCJ2JI94tzz2x3R+CmMfvUiYO9jfvvePvOCWTElti55J+BLXJN9VJjJkgm+zAwQJBJP1me+GHTc4yPr8SkrQSDcgNDfp0k3MGdxaMMcv8Hr5dVYCeDp1T206o/wDl9MMD8JUaenW7CTEmAPxMT39MDP1OOqFrJOO9A2Y+KKhY01qq9MwPEam4Pe3l1gA8RBgWIwprMKlYOzEam1OwUmTN/LpuxBPpxbFkT4foW/eR6TJPEiDMfTfttjpfhqmB5mYfa/YWmJ9f98J+bhHpASzSEeb6tUDjwlkIoSmwLKVUElQVY/N5jO4vY45brOY8h8NiUYMpnUQRMXMyb8zO5xYKPRKYggyNjLGB/wDESePviV+mqDZX+oP+mK/+go0lFAPNIrmZ61mKyMj0KxFvkC6rd4pnVf07bRgfJPXSm1MZeuC5fzFGA8y6eV4HrGLjRyQB+Yjmfr3wyo510PztHr/p9RiL+qpKuP7sD4u7Z5Z0tq1GvTMktJAB8l4I3JAB+uLD1XO1AyF2LhKqVgyGZZbgSbgbDb2nm/0+vDZifrf+WJS9Kpwh9wD+MHL+rJtPiSOWKVUeYVevrUzJrl9CFp06TAOkqpncEgmTHJwF1Xq6/tFSqpbU2nSw+UDToZWSxMqO/wDt6jnOkUKklqVJiRElFkjtO8enphXm/hykUamEUK4AKrYECdItBtJ5xcf6vjUrkn7fkSWZVVFMTqC5miKdVrK5KXgixA3/APdt64edL6qtFUSdQFgTAJmSBYn+/bB2e6W7otNoamvyqUpsBHbUpI+hxWM18Ii93Anbyx9oGGx9fgmqbpFr1MI7oi6KP2fNoxvqedUSAGBH0J1E3vcYb56nRoV6T02eKhqlpNgYU2HEk/gdsC5Lp703VnBqLsVJdbRFjqPmF7md9sC9a6ahAOXp1KZEDTIKkeuxmw/ONPzeOcr5rqvYuOXGlS97HWTo5eqr1HUeKHJQg6flMLIFiY5j37476F1AEVWZ/J4jBFW2kFizXENct34xRn8dJUBwO5g83+Ut7YH8aqkAERINheSdV5if9sMvkn9q/YOM4Xo9FzfUKNJoWsFZ2JbU4952NzM3ImZwxyhR5ZKqlDEECb3kc7W5748wXNz56jHxYZTqBC6SIHESJnAdN2gaasjeJK6Z9Ji/pg43FUpNFT4ydtJjeo+sAplYiflAvcmwiIkkWHA5vgsUs0U1rTSmsfMpYMLXBZW8oIneBxi3VOl0aWoFqVMmGIATUCY2A32F21c2HMSZ6mtmzLkowBLLqAs0KENDSDpgzNzAtfHP+M30v5Ol8FR/1FUT4WzFQCoNKruLMCed3uw22n3w2y/wp5dZrOd4ApkEkzBCjceq4f5fqOXqEpTanB3YUyizYeZS1gYF7cEYadJ6aFJbxw1UgnSvhuEkyoFGksERNzfyjFfEk9P+AXCMdoqbfCopjW1LMVG1ACBpkGJ1aiCAPYH05w5yXwcuoakEMQRMwR+oA64JBAOoqdxE7huHrGqX8VRoXyVGoIqzqumrSHuVSZC2AiTfDLwEqsKj11lQTamq0jMkMvjK4kHlGm1+wj5FWirVOmUKTuBQNQLpbSukt/CRJiVAvczB+x7V8vTWDWCMxiGUytxeNGlQARfYkCScPM/kg9MiXIP6iVqKbQfmZiAZ/TpPbCfK5YrXCVKFNqd2DUVrnTMH94joyHmw9yLzhfGynJGlqgfvA06iw1MzEEGDIZCQqWAImDF8DrnZM60hLEFgoI3nW8ypgbXFpOHOd6XRIYJlFbXGqKuidokArEAX52EXxBTyYy9PyU8zBI/9NaJILHks5hB3IUWwtwYtyRX6ee/aNTeDXkeYKA8t66EYA+k8RgjLUVsCDsSBHmBN21RMEnYsePuxzXjAPor0a1NtkAVWkzA8SlUUxcTAJ3iOY+gq1LShywpq19VCl/8AdqrF2m2y272xHFULciEZN2IhqiuRuujSdreGWMf84gTKwxQqzuAJqKVtuL6yVnuL8Ww4z9Jaakh2knUysiMW3lQHE/a1sD+MGPhhlBKkkKrCoBw2kJEd/KdxfCWtC5MRjMLBpwQwkABlb6k7g+sgXODFywKjymRGzUibCxMz/riVmWYZNRO7ayskECGDPJMcQRY+2M0QxZBHcAidgNlB7bCPacInroSyGjQ1oYMGb6oMG0ixk++IVpVA0FlAtp/eML32Bn7YnzNNaqqWZV2MAgz3EyCDFo7nGhokKAh9gD/Id+5/rhVi2D5oZgTcyb99uLj7TjdLMVSLreYOqO1zI2/5wRUpSuwUAXAGk37Q31xyuYYEB5gEbx/f1xXJNdIWwV81BjQjmeTfnuPcb98a/wASKNJoykAmDzO0Extvg3xNpp8TxO+wxHWba9h6b8i1/wCziJx6cf3FvRH/AIxQHmZWW8XEX7T7DjBuXzlNrpVMi0Fjb3k4CaqkHVpJMQCLdhY+l7d8aWlSYToUTO0zE73HviSjFrz/ACDyYz8QnZx6cj67YgOYe6sJIEysSY7CT64XnLrAPmBBPqNoHMxb8fTHP7I0aTUDDaDqWd/X0wKxx9/2K5MYJmZAmmQfUEW9ZAjEb1KZFx+DgIUGEQ88HzEdje3aIxLUzLAmCGIB3gkHYf364v4avRORpsvTbYjAGZ6ah4H8sS1c+LzSI9RA9AYiRiI1UNwWHt7i9v64fCM4+5LFOb6MvAOFVfooJk3PfD55AlawM/xjabxaO45OODqPzCfaD7742xy5I/8AYvnJeS5KGquq6qcMCQpRrxvLKNEj3+hwF1/JZrwpolPEBUnQAlRkA2Dh5J2uaYH5xI/W1QxqFRSYH7OqSpBAALOxG+r6Anvhh07PxIpIyu0lvFd59ylNfDjbdh2HGLha3SPV5N62Zl6lFKSHM0s1SaAG1BmUMSbaqYhrmJKaTMQbYIy/+EnUqmirMdjlwGJIn5alOWa5MR9McVqWcqgj9rWibSaVABgP4ZYuZtYh/wDTDOj8PgEOauZaqE0a/FanqWSxH7oiL3m5Fr4cpRX+DLJS8kXRwPCVHD1AgAaoadSjBA+ZaJWmQhnddXI74PbK0wZT9pchpCoa4B33LlU+pMYi6n01ZSqDVp1V5pO6s4gWYExV23YHbfAdCvWRjVzGdSjS0H9watI7frNUBWU3AIWbncyMEqe0Lk2jvqfQjmgEqZehQpk3LBalYiCIEDTTPrqe098QdF+HkyqhcrXrsrN5m10ioIEElzTMjy6YEwYtyI2+LMgSyjMUzG5qM7qf8qF7tafk25ubhDrXTy3il6r6CSuijWCJYAxoQAkROpiSNRuBYRqXWwGyyLRcn965YReA3/2SAeOBhb1HpWXchgteQbaHzCn1N2A3+vGE/wD+dZVnCK2ZnvBgHiQTqHHHOB+r9YzhaKAApb+KVngmIcWgiJAgyL4BpoW9nPVeqOpRUq5qj5wNeZ1BSORTPhMpaO5++HGaqwAyVq2qIE0iQT3ISkCT7Ec4rn7ZoId0eqwBIatUUNN7KgkAATcbX9cGdL63WqqGWgIn53raQfppEqAQLYXJ2ugXoLy9WoXKGrDMCQAKqnT+o6bOfoALgWtiHJ9Hq0iTqp1QTINRGLLErAeSRaLfgYkzXUahiRllI5IaqbniI9B9MBKC5GrNgSLKiCme8AiDt3wvpaBk7GTVat5FIE2Eam9AL6Ym+2Fmb8bVBQNJM+G0cAyVYb23nkTiLM5BQ8rWqAxw5kn1I7Se8Yjq0WnS9d4Fu35nzH39cKcEtsTNm6eepUpFgbySskSTyO1xHoMFJ1BGuzXIGyEReOf6YCp5OmsFVBm0loJvHaBMT98C/wCHAltRKybNzttJ5v8AgYFwhIU2HPnlkNLQdp1bR+LsbkcRiE51yZQsVXy8cD8X5PriIZNChkkmSbDgETsRODWMKRZZ8rHciwsIkiSO0zztimoLrYDZz+0M589iRIOwnt6Nf8d8S02ZQP1E3uTt62vvue572jp1EDEBZj/+Rfi157zAwOlRAJibAQCCI/SNv4RfcEx64DjfgpoNQ07DSp9JPuI9N7+hxGaiEA6YAJjjiST3vbftG2BGzesiG06ZvpExpJNvY7T/ALcvWA0gtpBEwBcWAKwN5BJEb4tY2LaDMuqjSdR80QLgbMBv6wTfGtNRjaoCTcCLen0P9cKDWYNDEFYLA94EEdhyJHa98F5PPSVXSZaIiTaYO0X3H+mCliktrZAovVW/ho0QPfaTHpOA2zgmXpspJ7TO5Hfv+cEU8xpBYkhYE9ybwJ5J37geuI6tVtMnzcEmDxF7Sbn2/rIr3RDBm6Z2qACSIj6R9/x64FrAElQoYntI529feOMd1K4VQrU9iDsLmDYj3nnv3xFWKQHIgTYc7eaR9/vg4xp6slAeYy6/5htsREC0XG232GBa+WYHysQPr/TBbwT8zRFr3ifTftvgVt5jUuwJY8Ezt7j8Y0xbIrPQMhkuosiyuVpgEkBjWq6RHOtz5r7zhgMlngzDXkgeSaNaSALn/wBQxvxgapnaFVmJRapU6WstovDTBI5jEWW6Z4YYU3NJJ1Qj+K07wviKyqLm0HjE+J9EvyPVvG67f6jdcjmmt+2hLbUKKKPU/vC5A4kRt62Rdd+Fa1Q+frVVE4WoQDMAbLUSfbTafXBb5jMHyU2q1pE6XreAY5I0UVbfiR7Rg3JNWe/h01bY6XNUkjg1CQRaTed8HHJKO1/YTLCnplDqfBuT1+HVz1Wu67rQoO1QSZhzDhYnk3n0wd0T4EyRXU9Ku2owvikLpF7lE0QbH9bC4Nxix9UfM5eoH00qutvDgHw9NwEZvEqHxGaIlQCAAPNsHVOoYECpTaRqDpUcCwmNLBVO17xcRvhks0vDFfDj5FWXzGTpU1pU/BCqFCklDbZQziZa2/8AZbZXOtcFGK8MpVlMkgQFOv7rGI61By+oVjAiVNNSIEyReQSSJJ/htEklT8VVKyUjVpXcC58QoALFtMgpJgWP0vbCk7YE1obZmpTF7rFpCOLe4EEW74UZrrtCmTTDmrUmAiqWfbkKo0/W2POsx1HM16+jTU8QC4DEleQdWyC86hAPM4tvS/hugKY8dQ9QiWYat7ky9pN/TBShx7M7dgdXpVXM1lrsjUYMzcGBsApuD6z3w7Jg31W2BZjb/wAj745fptMIfCZ9QmJrVYtAAjUZtPH0wBQcioyl3popgB1QEneQ4WNPHvN8KbtAvQYuXpAyKagkROiOPaPXHVUx62teNj7zb/T6YaRC/OxX6bepIJwCxp9jfvPO39xhe2LbNtTfcs0SLAJHHqW/OBc0jknY2vNpt6fT74mbPqNrb72Fjaee/GBcxXdgBe5Bn+9hEnC7lYqbYPl6MsPLA+oFpE/zP1wUaJBkQSAdvaxG3H88D5PMsCoqL3Eni2/GOGquxY7C3sL2kzb/AGxTUmxDDWB+U2HdY43EDiBxv6k44zYJBQi4WY1KwMeZ5UjVsLGd4HcYXNm6gULpkLvyu8mIt6TiWp1YFgy2UwpiIEC8jmSswN5wccclugbDcrSJY6g2nTCmxJJ1FZiDB0Mt9O4McYV1zULsFHhrB07zIOkH0JDcAC2G9HPKDOq2oATzANiPQn0B+s4irmmrLAgsQYJBIsWMxeZn6Qb2OJGcl4GKToX5Sk95XSQJpjvI0qA19Py353xrMUHTSFILKYM7KCLCLfm8nkRLF80FJbUIUebzCTuNQvwRAnAmW6jTDIx5DTIMwolWi0iYgmYj0walJu1EBu/AJk6ko2mYUOQx9ApMDgQ4MG/l9MZlXDNeVRAgJ3MkFoJ7gSIHfB1BxpdEJUDyhQNyWChh6+abngdsS5SsBUYs60w40jRfy6vMxYXiSSTtPeL25d6B7IauZAhtV2ixEaZAWIBM2Hp7XvC2fCsJgsTK91mYmNvpgnM0qYIYENEgfMGAgbrwSvY2kxgPPgEfu9IYi6ySQAWEAHe6g7/bfAwUX4JRJVzSiFb5lJuLx5gQb/NJEyTz6jEK5kA6YkQWtIvYAQNri8zzgKtRgqPMCoA2uQLxHB+b3t2xzXLE8CLRubH19Afx2OGqES6OquYgRALLuQb8TNr/ANnA/jAE2uTOxv8Ab1nHVbMMpYU78SQBfa3rvEn+UAJc8VMsxvH4Ht6jDowbXQSR6B0frGX0qKGUqqDZWZXYaQf4lkcGPNF/phrl+oGodKUqThdUgOXG8QToswjYRzfbDDKvBks9Vt7AR6EEf36nCzqHUaT1NASpUqaZOiUsSwvUlP4TYt3xlbjJ2ketqUVTY8XMsi6iaVDiCmoA951JxvvxgB/jXLCpobPUweyqI2E+fUyAH74rfUOipXQnM06iKGVgEqo7LAIKu7AQCCDPm2sRsS+m/BOQDBlpM8dyWUyBeDYxf0vzGGR+Gl9pv8qESjNvSX5j6h13L6Eq1cxTLCdJ1qR+kkIFidh3Nrm2AOr/ABbQoeZcrWZiAdQpMobe2the43vvhxQyFCnU8VaaqSsMIAFyCJWIBtE7xOGVKqkkIRztxvO3H+/bEUo30BNNI8/b4h6nXYpSywpLExUYLUIsARMEML7gjvtepZjpOfq1fCzNV1U3Jq1tS2NpVGIn7c3x6z1jq5o0mcXJmDIAFpmdXf8AnhJ8NZ1K1BWqKS7gMxLWJHkLKQZC+Ux2H1w5ZeKtJGaUb7O+i9O/Z00ApECSUOpiBux8Q6uB/sLbz7abEiBe0kxuRbbj8Y3VyVPTAD9h+9qFfT9R2Hp7YXZ3Ko66ZGn3P0Mkagd+bW9sKtyfYDoJqZmVhIjv/oLT2ib7WxCquSPMzQBMhRN9rLb74AoGvTRqUF1X5GZlFoEKYEyJidN+caXPuPmpaQBqkNItciCAfqbYt42l2JlQS9JwLs5kxGojtPygfUXwO+URY/dmYI+ZifqSZNsD0eqvUAiBqusyxAubxHAHO2IcxUqeYFyCRpgCIESZJJO9/tgXCV0KkglyiCANINrd4gySZ2j84ysocTJkj2Jva59j674XCibE3IBjuRCwNRvNyO9scUC4YXkGTME3ixg8XHbY4GWPymLaDcwTAadxJXtfT/r9hgPL0y2omwLje0gXj0kDc7Tjh6hlgAdMkSdhBMA3ttO/IxvKUiTBJmGkHYGGgnn9QB9RiKPGL2KaD89lQVVlAJDQIi8QYPtMc4XVKENYMJIYxGkKdxO5Gx1HsALYZ0qrIANVzYjvfV7AD+h74jreZSJhyJkDckTz+kQRItf7BCUo67QFAFB3FQJcNIaRPpsw2AgieJO+D6HT2WmSQSwXxGZDJuASyru6A9hEmTAjEGWyVRQrKousBpmQATAP/wCv8u2JxR0sp1MoRhpIYqYWGJ1bLZfpftg3JX9CtCxCroryQIgsQSXvq1ADiYkk7qYG2JWygYqakBAiM+kGGkhnA2mLr6CD2wTSrF9OsyxErcCSQrs0ASN5sR8w9MClahLzqZSRdQQIIJFzfm//ALe4gs5b9iXoloHxWdn1BmksALpILAKoIm0TPtN4x3++KrTVNQ0MrOA0WBEa4kGCRHE7Wx22X1C3iBzBZfmYsX07n9OlpF9u95x6SBtJJZgTpAleNfmVzrLmDt62gYG7shn7GFjXfbUYmPIJOnVECOSDbANRYCpqDFbkqxIBkzpsLxE2573wHLHUzyEL7AnUxAP1Eloj1+x7o7goFBUXupBWAJAYTAJJA1Wk2wfGu2TogaqFUgqwI1SpOkcECCDMg9/ziDMZ4QdM3EyYkT5gL7WI+sTgzK1SwCOwi8KbxBvFiZB8sC8n0wvz2XqEeGNWmTwYnYTAgxBuTgoqLdMJJeTipmlYkXOoidhcWA2sACfucDVZY6gQk8S39Bjo5EgwJLbSY7kMYm1wRHPE74H6nIIEXAvvMwNyI+3H3JfGKukWkro9drZbK0iWLuWAGqatSIjbTr0CwI2G5wxyWXo6FNOmEpm4AAUHUN7byfod+8LqPQkBbW1aqSFB/eui22AClSYABi4k4Ip5dKRQeeXMQPMAALa2Yzz3neAd8cxyT1dnrOPnoLLkkFTvaBxc8Ab89xiaizL5CLWEqAJ3n9Rj2jGywKyDA3kki07QRJAHt7c4lFWCVMAz94AvERFxt9TgEW3olMW0ybzf32vcb4GqZvSIJjedjPYE7DnBQf2Ijedx6RFj/XAlV48sQOIHeSY+208+mGRM8yofGfxEpp1qAjVoJYMVIK6WjSwtqBAI7HTycV34a61maKLTp5RiFBMnUskzLFnAAJJ2HbbF7q0aNRSj01uPMp3gNPzAyRMG+BamfoUwSXLfQkj08oIPPONkZx48aMc4u7Eb9dqNBKqBJBSkdbiym6wDEm+mdxbBn7Q6yvgVODMAT+eBgPqPxrQRgEk2gjuIJCkd5A9sVnqPWK2fqCnSU00E6hJiO7xsLRpvhix34pCJS8D/ADXxTTiCGDCZEDePfv8A32W5vqjuCvhutI/Mb35EGBFxNu2GHTslRoKFChj3cAmdp9Pbi+Cq+cC3BJ3JMwNoibn8YByp0kKdgHSK9KoFAnygDzDa1m95nDSs6sDtJM+v9kavvisdYBSp4idtladifm2P/GIR1mppLMo0j/KRxIv9/v8AanBz2hbHtesOAIIn13Bt94PqBjhBGnUATcniDeebiBz/AEwl/wAaE2U2B8tuYJEnYbcXjjEL9buGKmBI35IgbD2t/wAYr4MmqQDTLBVzAE7SRcDYmYtHMSZtsBtgtdIJZtyIX7gEmN9hfsMVOjnwCCBaSJHc3jf10/UdsFnqTK2kgMCPW1xsSew29cLngfSFyQ6NRGUvYsqkDi+sjg3+W54mZxFVzAdF1MWOgaioJEmTpkDysV1ACLR6YSVDfSSRDFdYiSAZ2MfqP4xmXRVp1CT5pgTMcjaRJggzeNIwSxL3B4jStm9GlBUktpB0x5exif4iD7LhlXmpSSofm0sTaCRpC6igEkG51ETwTa6nI1B4gepYsbrABC6ipMx5WVriR7+WYjpo7VANJGh5JuAUhwG48oENG8G04p4tfyDx0EZSoWK6gSkwBIGwEkdjpi+0n0sScyyVHOsXv5R86wTTKj9SwdV7RA3kYRf4mUqnXemgUNbVtAAkMPNJMSdzJ2x1Vz1N00rqWDqpu7SxUkB6RfhpGoE2BVhs2D+A32tFqDob5frC+G2oNqBKjRpAIZrLqMzGoxbZTe+BzTQqSqmWdBSpyXO7CJ1A7lYJ2JMi1waJrVdbU20sF1RTchmCzJ3uwUzpMSAxHAMVPqBUzSQtUKmWIJKgW8sbEXltyZiOb+FXREqG4oIhhSSQrEtcyfl/T8zkgmQNhzzDSY06Z8QNoa7ML69O0ciWaCT9LjEZr0/Cp1CQS3zLIaDH8YJIEDbceIfpNnM4rU7KFJA8h1LBUAWdpEwQYMC4MngeMrBUWSZnMuBCQv8AHABJJIJ8jXJgk6hczvwFjdUSSPCFRlF2diNiSDCRtO14xG+aJQqG2AU6SAwQENYQRY2sbgKcKq5UOvhnUOxXTEXOq5v99vph2PEvIxRDM3mbhmAkNHlACzuBC/6XgxscF0OtHSAVVo7mot+T5ImbXN7fdQzoGFlY/qLX+0yCPpidaywCtQ053UCmdufmX+WGPGqWguJ7fkc4KrMEZYXsSSZkTtBurWvgiqukwSRcbSTz5T2E3ntxGEmZ6SrQKlfMxMStVl4mYUeYfynGunfBuXpMXJashJgVm1gAgz5QQp838SnvIxykoVd/seok5J9Dxc0kkE6isMYOogRMnSDa0f8AIxC+eAsvlMn5gQFmT8phgfUiwM98Kc70fp2ktUoIigElgRTgXWQVIEeYi8XOxMY806hmalKoaWWp1g4YMLs4ZSFNMmkwbzdnEbxAw7HgWT7rEzy8NyX+/geqZ/qnhpqqVhS/hWo6qrRBkHzarAHk34OKX1j41pGt5qpq07gqmoBSDqsRp8RTtExfkYrVD4Uz2YqTVHhzBLVTpiwA8nzDygACNgBbDgf9PqdJlNWtqBuoUaNUcSSTMX2Gx2xpjixQ+9K39BEsmSf3Y0vqD9e+PajKgy37tRZiUXUeFBB1KFjaO3pdd0/J5zNq1Q1XCEx+sBjHCoII4JMYtmUyeUpT4SU9UAyVdjbmWLFTPFtsTZuqpvAEkCSTtMQqsfe2DWWMdQj+omWKTdyZXch8LUkIZ2NbaF06V++q8QZn7HDhnCfp0iBYAaTFzeOPbnENbMSukWURKsASLxxYgx3+2OK45BA7RPA9xf374BycvvMW4pdHWoH0tO0jff8AJwO0dgRxfe8i/tfGqmoi03soj2JkYDqIUgEieDIta4P3/OIkLYTWqmRIkG8Djk+2B6lPWSSxGkbATebDbcziNKoA5DRBieSREYjo53gJb0mbSYIJMj0OD4tdCmcU8uoJMeYknUTIgWmy8HGxll1qTTADCOY3399jJtbGJVvULi9o1WA/Tcb/ACwB2jA9XMQQtmgkXmJFoBH88F9q9AMmqZRdUADcwB9CCO1wDzExjbrogMQRJtIkGDFz6gDEP7SpBKMU0i6mTq21ReRaTB7G/aBMyplGvY/NHaBf7H64pxk+wGho1QMhJJGm+3E6Zjj5j9cc/wCIaVYqVDeVQNMyAZJANg2wne55OFy5iNhsmlr8EWj2OkzOOhSqaSFBgX0bxb5o3sJ9pOIsaXZKoedVFauFqsyt4aEVKgiQqkkOQCZYkMpm5df8wJUVq7IWYOfLAUjVB1AwwYCIjywYsfTEGXzukkAwrKJtFyoG/I1DGs/VqBFV2lSzFBO1yrflZ978nBxjumV5I82wCrKwHAtO0E6TF7RMT/THOToO5MNBQAxcER2AHmvEf8YEIJAIj5TPc3LT7+b8YkpVDaxOkcRIAEEyNhHf+d8NqloJrQdRYcM8Kw3QIywZYAAmDO1+BgvLdfdDZiLGEkFRYgAkASRPrccbYFObuC+ptJafNcggCA5B2LAxfAFPNA1BpCoGIkqoMA7kSJBAnaJjC+CldoBR5bodVcxVuKjhlJEAVFM8jzAn1EcX5x2KTM5VatMrUWVIVxeGA8rCxEuhJkXO52V51tLFrBSWNMQG8u4BuCbEbzjdOmN1qEqdJeBdbgEi0xBge8Hia4a0TiQhxMtIgmW7mCSCIsYtHv7Y1lao1XGqSRbteLESQfofbAmZqksWEgE7A/Ue/ee5xisx41bweRfeQZBtzbDuOhnHQcFDKdDD5gRSYSSIksjkdwRFj74gNWmbxHo1/sQP5jtvgZwbSZPuJn6TjpFBJLMQZ+p9774ukXSPcch8U0XAYLAY2nUZj39BuY32PNf6v/1Np0wyUqQLQRqAAAMHTcqC0GARpHMHGYzHOwYISm0/B3vUzcIKS7Kvkvitsy5XMu4U2RKcKoYkRM6oEix0sQT2xYfhHpSUtRFQHMNEqGqABCoKqZQo8ggmdpEEEYzGYZ6pLGmo60L9H/y7nvY2zecqAilPnYgU01QdpIb92UChTqkGSARvAwLnOlFlJepAVl3XWUZRrXS/lMBQw2MhotjMZjNH7NUPyO+wTNUyCQyq5IAk+UkEqP0yJ43vFyMQM604lYkmxM+YAzcA7/674zGYZDozZOzl6w2CyJ9B97b2xHTzOlQCbSd++8bH0xmMwaMkyJ81K2G5kE3gX1QJi8nfAburCB9TfewDCTe02tucZjMWhD7Oa1NDJuJm4uJnsfr98C1mOkGLFjH5/wB/tjMZgo+BTBa2XIOkbG/aRqgfa5xGFPmcNcXna8TP3vjWMw1NlApVjBmSb/b/AGGJlqmRMEdh6za9sZjMMZTJKVMK8vcamED0E/1/GN165mVJkAMItER/Q4zGYBK2CAGsTJ7xI4sZxLnswzIgMQJiPz/PGsZh1bQT7BUMwJ9PvIxmUuyiSCbAj1O0cb4zGYLwwvcNrFqbTAgiYNwQR2+mNVHptBUGmd9MlhE3gm/GxnGYzC49C10SZXOULGp4wYbaCkfYj2wJVqDX5CwvzH9LdrbYzGYLig0kZUpiA6iLSV9JK6ge2obG49RiGov8J8pMQdx78fbGYzFloGAwTTf/ADHGYzBMtn//2Q=='
        st.write(' ')
        st.write(' ')
        st.image(url, width= 300)

    with col2:
        st.write('## Climate Change')
        st.write('Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns')

with overview:
    col1, col2 = st.columns(2)
    with col1:
        st.write('## Impact')
        st.write('Effects on weather. Global warming leads to an increase in extreme weather events such as heat waves, droughts, cyclones, blizzards and rainstorms. Such events will continue to occur more often and with greater intensity')

    with col2:
        leaf = 'data:image/jpeg;base64,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'
        st.write(' ')
        st.write(' ')
        st.image(leaf, width = 300)
        #st.button('Image 1')

with datadesc:

    # Loading the dataset
    #def get_data(dataset_name):
    data = pd.read_csv("E:\pythonProject1\streamlit\Clean classified data_kwisha2.csv" , delimiter= ',')
    X = data['clean_text']
    y = data['class']
        #return X, y
    #X, y = get_data(data)

    #Data description

    st.write('## *Data used to develop the model*')
    st.write('The data used for training this model were scrapped tweeets from various African countries that relate to climate change and the impacts the African continent has been suffering from due to the negative climatic changes happening.')
    st.write('\nNumber of tweets collected: ', len(X))
    st.write('Dataset classes : ', y.unique().tolist())
    #Preview the data
    st.write('sample data')
    st.write(data.sample(5))
    #Exploratory analysis
    # col1, col2 = st.columns(2)
    # with col1:

    #     #st.image('wordcloud')
    #     #st.button('Image 1')
    # with col2:
    #     #st.image('wordcloud')
    #     #st.button('Image 1')


with model_prediction:
    st.title('Classification Algorithms')
    models = st.sidebar.selectbox('Select Classification Model', ('Naive Bayes','KNN', 'Logistic'))
    #dataset = st.sidebar.selectbox('Select dataset', ("African_tweets_dataset",'African_tweets_dataset' ))
    st.write('Classification model : ',models)
    #st.write('Dataset : ',dataset)



#Adding model parameters
    def model_params(clf_name):
        params = dict()
        if models == 'KNN':
            k = st.sidebar.slider('Number of K-folds', 1,10)
            params['K'] = k
        elif models == 'Naive Bayes':
            alpha = st.sidebar.selectbox('select value of alpha',(0.001,0.01,1))
            params['alpha'] = alpha
        else:
            solver = st.sidebar.selectbox('Select solver parameter', ('liblinear','lbfgs','saga' ))
            params['solver'] = solver

        return params

    params = model_params(models)

   # def get_classifier(models, params):
    # classfication
    from sklearn.model_selection import train_test_split

    le = LE()
    y_encode = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encode, test_size=0.2, random_state=50)
    # vectorization
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, )
    X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)

    #Modeling
    from sklearn.metrics import accuracy_score
    if models == 'KNN':
        clf = knc(leaf_size=17, n_neighbors = params['K'], p=1, metric='euclidean')
        clf.fit(X_train_vectors_tfidf, y_train)
        y_pred = clf.predict(X_val_vectors_tfidf)
        #st.write(f'classifier =  {models}')
        st.write('*Confusion Matrix*')
        st.write(confusion_matrix(y_val, y_pred))
        st.write('Accuracy score', accuracy_score(y_val, y_pred))
        st.write('\nClassification Report')
        st.write(classification_report(y_val, y_pred))


    elif models == 'Naive Bayes':
        clf = MNB(alpha= params['alpha'])
        clf.fit(X_train_vectors_tfidf, y_train)
        y_pred = clf.predict(X_val_vectors_tfidf)
        #st.write(f'Alpha = ', alpha)
        st.write('*Confusion Matrix*')
        st.write(confusion_matrix(y_val, y_pred))
        st.write('Accuracy score', accuracy_score(y_val, y_pred))
        st.write('\nClassification Report')
        st.write(classification_report(y_val, y_pred))

    else:
        clf = LogisticRegression(solver=params['solver'], C=10, penalty='l2')
        clf.fit(X_train_vectors_tfidf, y_train)
        y_predict = clf.predict(X_val_vectors_tfidf)
        #st.write(f'solver = ', solver)
        st.write('*Confusion Matrix*')
        st.write(confusion_matrix(y_val, y_predict))
        st.write('Accuracy score', accuracy_score(y_val, y_predict))
        st.write('\nClassification Report')
        st.write(classification_report(y_val, y_pred))

    #Testing the model with user input
    st.write(' ')
    st.write(' ')
    st.write('__*Testing the model*__')
    st.write(' ')
    st.write('Testing with a single sentence')
    sentence = st.text_input('Input your sentence here:')

    if sentence:

        sentence = finalpreprocess(sentence)
        sentence = [sentence]
        sentence = tfidf_vectorizer.transform(sentence)
        y_prdctd = clf.predict(sentence)
        actual_y = le.inverse_transform(y_prdctd)
        actual_y = [actual_y]
        st.write('Your text was classified as : ', actual_y)

    #User uploaded dataframe
    st.write(' ')
    st.write('__*Predicting a dataframe*__')
    st.write(' ')
    spectra = st.file_uploader("upload csv file", type={"csv"})
    st.write("File MUST have column named 'text'. ")
    if spectra is not None:
        spectra_df = pd.read_csv(spectra)
        st.write(spectra_df)
    #Prepocessing
        spectra_df['clean_text'] = spectra_df['text'].apply(lambda x: finalpreprocess(x))
        X_test = spectra_df['clean_text']

    # vectorization
        #tfidf_vectorizer = TfidfVectorizer(use_idf=True, )
        #X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X)
        X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)
        y_pred_df = clf.predict(X_val_vectors_tfidf)
        actual_y_df = le.inverse_transform(y_pred_df)
        st.write('Your text was classified as : ', actual_y_df)

with endnote:
    st.write(' ')
    st.write(' ')
    st.write('****'*30)
    st.write(' ')
    st.write(' ')
    st.write('__*Key Take Away...*__')
    #st.write('__*We cannot burn our way into the future. We cannot pretend the danger does not exist, or dismiss it because it affects someone else*__.')
    col1, col2 = st.columns(2)
    with col2:
        st.write(' ')
        st.write(' ')
        #st.write('# Impact')
        st.write('__*We cannot burn our way into the future. We cannot pretend the danger does not exist, or dismiss it because it affects someone else*__.')
    with col1:
        st.write(' ')
        st.write(' ')
        leaf ='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZSiq5QG6wi8nna1E9rTamMzGvLKab_ebOnA&usqp=CAU'
        st.image(leaf, width=250)
        # st.button('Image 1')

# The easiest way to start a conversation about climate change is to say, “Hey, have you thought much about climate change?
# #Second section of the app
# with desc:
#     st.title('Overview')
#     st.write('This is the overview of the app')
#
# #ploting the dataset
#
# from sklearn.decomposition import PCA
# pca = PCA(2)
# X_projected = pca.fit_transform(X)
#
#
# #The third section
# #with plot:
#
#
# #The input and prediction section
# #with prediction: