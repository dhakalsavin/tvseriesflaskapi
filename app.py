from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Load series data from CSV
series_df = pd.read_csv('static/app.csv')

series_df.dropna(inplace=True)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Extract features from genres, titles, actors, and directors
features = series_df['Genre'] + ' ' + series_df['Title'] + ' ' + series_df['Director'] + ' ' + series_df['Tags']
feature_matrix = vectorizer.fit_transform(features)

@app.route('/', methods=['GET', 'POST'])
def receive_data():
    if request.method == 'POST':
        data = request.get_json()

        if data:
            names = []
            genres = []
            tags = []

            for show_data in data:
                names.append(show_data.get('name', ''))
                genres.extend(show_data.get('genres', []))  # Use extend instead of append
                tags.append(show_data.get('tags', ''))
                actors=  ['Joaquin Phoenix', 'Robert De Niro', 'Zazie Beetz', 'Frances Conroy']


            user_preferences = {
                'Genres': genres,
                'Actors': actors,
                'Titles': names,
                'Tags': tags,
            }

            # Create user profile
            user_profile = ' '.join(user_preferences['Genres'])  + \
                           ' ' + ' '.join(user_preferences['Titles']) + ' ' + ' '.join(user_preferences['Tags'])
            user_profile_matrix = vectorizer.transform([user_profile])

            # Calculate similarity scores
            similarity_scores = cosine_similarity(user_profile_matrix, feature_matrix)

            # Get recommendations
            recommendations = []
            for i, score in enumerate(similarity_scores[0]):
                series_title = series_df.iloc[i]['Title']
                series_img = series_df.iloc[i]['Image']
              
                recommendations.append({'Title': series_title, 'Similarity Score': score, 'Images': series_img})

            # Sort recommendations by similarity score
            recommendations = sorted(recommendations, key=lambda x: x['Similarity Score'], reverse=True)
           

            data = recommendations[:10]


            print("The top recommended Sereis are\n")

            for i in data:
                print(i['Title']);
            

            print("The similarity scores for recommended sereis are\n")

            for i in data:
                print (i['Similarity Score'])
                
                


            print (data)

            # Return recommendations as JSON
            return jsonify(recommendations[:10])

    return jsonify({'message': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)