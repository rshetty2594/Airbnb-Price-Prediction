from utils import preprocess_data, get_tf_idf_mapping, \
                    get_cosine_recommendations, get_dataframe_from_csv

df_listings = listings="""SELECT * FROM [mineral-aegis-248102.Project1.listings]"""
airbnb_listing_filename = gbq.read_gbq(listings,project_id="mineral-aegis-248102.Project1",dialect="legacy")

location_listing_df, location_id_mapping, location_id_mapping_reverse = get_dataframe_from_csv(airbnb_listing_filename)
cleaned_location_listing_df = preprocess_data(location_listing_df)
tfidf_matrix = get_tf_idf_mapping(cleaned_location_listing_df)

def get_discount_item(item):
    name = item['content'].split(' // ')[0]
    description = item['content'].split(' // ')[1][0:165] 
    discount = 0

    if (item.price > 500):
        discount = str(item.price)+'discount of 20%'

    elif(item.price>=300 and item.price<500):
        discount = str(item.price) + 'discount of 15%'

    elif (item.price>=150 and item.price<300):
        discount = str(item.price) + 'discount of 10%'

    elif (item.price>=150 and item.price<300):
        discount = str(item.price) + 'discount of 10%'
        
    elif(item.price<150):
        discount=str(item.price)+ 'discount of 5%'

    return {
        'id': int(item['id']),
        'name': name,
        'description': description,
        'discount': discount
    }


def get_top_airbnb_location_recommendations(location_id, num_recommendations=5):
    all_recommendations = get_cosine_recommendations(location_id, location_id_mapping, location_id_mapping_reverse, tfidf_matrix)
    top_location_recommendation_ids = all_recommendations[:num_recommendations]

    result = []
    for location_id in top_location_recommendation_ids:
        item = cleaned_location_listing_df.loc[cleaned_location_listing_df['id'] == location_id].iloc[0]
        result.append(get_discount_item(item))

    return result
