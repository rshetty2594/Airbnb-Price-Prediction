import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def get_dataframe_from_csv(filename):
	df = pd.read_csv(filename)

	location_ids = list(df.id)
	location_id_mapping = {}
	location_id_mapping_reverse = {}

	idx = 0
	for location_id in location_ids:
		location_id_mapping[location_id] = idx 
		location_id_mapping_reverse[idx] = location_id
		idx += 1

	return df, location_id_mapping, location_id_mapping_reverse


def load_model(filepath):
    file_content = open(filepath,'rb')
    return pickle.load(file_content)


def preprocess_data(listing_df):
    listing_df['price'] = listing_df.price.fillna(listing_df.price.mean)
    listing_df['price'] = pd.to_numeric(listing_df['price'].apply(lambda x: str(x).replace('$', '')), errors='coerce')
    listing_df = listing_df[['id', 'name', 'description', 'price']] 

    listing_df['content'] = listing_df[['name', 'description']].apply(lambda x: ' // '.join(x), axis = 1) 
    listing_df['content'].fillna('Null', inplace = True)
    return listing_df


def get_tf_idf_mapping(listing_df):
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
    tfidf_matrix = tf.fit_transform(listing_df['content'])
    return tfidf_matrix


def get_cosine_recommendations(location_id, location_id_mapping, location_id_mapping_reverse, tfidf_matrix):
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    location_distance_vector = cosine_similarities[location_id_mapping[location_id]]

    #Iterate through each item's similar items and store the 100 most-similar!
    top_indices = location_distance_vector.argsort()[::-1][:100]
    top_location_recommendation_ids = [location_id_mapping_reverse[idx] for idx in top_indices]
    return top_location_recommendation_ids[1:]

