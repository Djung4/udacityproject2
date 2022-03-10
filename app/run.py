import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# functions
def tokenize(text):
    ''' splits the text into distinct tokens
    Paramters:
    - text: text that should be tokenized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens

# load data and model
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
model = joblib.load("../models/classifier.pkl")

# flask paths
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # category data for plotting
    categories =  df[df.columns[4:]]
    cate_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)
    
    # plotting categories
    direct_cate = df[df.genre == 'direct']
    direct_cate_counts = (direct_cate.mean()*direct_cate.shape[0]).sort_values(ascending=False)
    direct_cate_names = list(direct_cate_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # category plotting
        {
            'data': [
                Bar(
                    x=cate_names,
                    y=cate_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },
        # category distribution
        {
            'data': [
                Bar(
                    x=direct_cate_names,
                    y=direct_cate_counts
                )
            ],

            'layout': {
                'title': 'Categories distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()
