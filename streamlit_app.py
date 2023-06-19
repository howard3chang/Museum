import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium
import geopy
from geopy.geocoders import Nominatim
import requests
import json
import edgedb
import string
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
from scipy.spatial.distance import cosine


import openai

openai.api_key = "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"  # there is a limit on this API - if used too much it will trigger shut down

st.title("MSDS 459 Museum Recommender")

st.subheader("This application is to better search for museums that fit user preferences.")

st.text("sample address- note for demonstration San Francisco has the best results")
st.text("550 37th Ave, San Francisco, CA 9412")
st.text("600 Guerrero St, San Francisco, CA 94110")
st.text("11301 Bellagio Rd, Los Angeles, CA 90049")

address_query = st.text_input('seach address query- please  write  in  format  of  street,  city,  state, zip  code')

st.text("sample I like outdoor scenery and landscapes")
interest_query = st.text_input('interest query- please input what topics you are interested in')
button = st.button("Submit")

# Wait for the button to be clicked
if button:
    # Process the user input
    st.write("You entered:", address_query, interest_query)


# Foursquare- to start

CLIENT_ID = 'GYE35LWRDA535NWYTHAIW5R05ZMK4SH2IRCERZTOXMUN2MT1' # your Foursquare ID
CLIENT_SECRET = '3IQEHJJ4EYZGWD5KIAT3NHOLNTKTMVWA4T5SZSOKHZGH0TKY' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

LIMIT = 70 # A default Foursquare API limit value

address = address_query

##replacement code- use google geo instead

GOOGLE_API_KEY = 'AIzaSyCy9hYptxqt6A72XBlT_jcm6tzVPelD36I' 

def extract_lat_long_via_address(address_or_zipcode):
    lat, lng = None, None
    api_key = GOOGLE_API_KEY
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
    # see how our endpoint includes our API key? Yes this is yet another reason to restrict the key
    r = requests.get(endpoint)
    if r.status_code not in range(200, 299):
        return None, None
    try:
        '''
        This try block incase any of our inputs are invalid. This is done instead
        of actually writing out handlers for all kinds of responses.
        '''
        results = r.json()['results'][0]
        lat = results['geometry']['location']['lat']
        lng = results['geometry']['location']['lng']
        latitude = lat
        longitude = lng
##
    except:
        pass
    return lat, lng

latitude, longitude = extract_lat_long_via_address(address)




search_query = 'Museum'
radius = 8000 #Radius of search in meters

url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
results = requests.get(url).json()

# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a pandas dataframe

dataframe = pd.json_normalize(venues)

#dataframe.head(70) santiy check

# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

#this is to make sure the venues are Museums
mask = dataframe_filtered['categories'].apply(lambda x: isinstance(x, str) and ('museum' in x.lower()))
filtered_df = dataframe_filtered[mask]

#there are four museum categories this is to filter down to just ART - some general museums are art too but for this example
Art_Museum_df = filtered_df[filtered_df['categories'] == 'Art Museum']

#get rid of duplicate venues
Art_Museum_df['nan_counts'] = Art_Museum_df.drop('address', axis=1).isna().sum(axis=1)
deduplicated_df = Art_Museum_df.drop_duplicates(subset='address')
candidate_venues_with_lat_lng = deduplicated_df[['name', 'lat', 'lng']].drop_duplicates(subset='name')
candidate_venues = deduplicated_df['name'].unique()

#add map
museum_map = folium.Map(location=[latitude, longitude], zoom_start=16) # generate map centred around the Search location

# add a red circle marker to represent search location
folium.features.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Search Location',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(museum_map)

# add candidate venues to the map
for lat, lng, label in zip(candidate_venues_with_lat_lng.lat, candidate_venues_with_lat_lng.lng, candidate_venues_with_lat_lng.name):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(museum_map)

#display map
st_data = st_folium(museum_map, width=725)


#display the candidate dataframe
st.text("inital candidate venues")
st.dataframe(candidate_venues_with_lat_lng)

# Create an EdgeDB client
client = edgedb.create_client()

# Execute a query
query = 'SELECT Museum { name, exhibits, exhibit_description }'  # this is actually exhibits
result = client.query(query)

query1 = 'SELECT Review { museum, text}'
result1 = client.query(query1)

query2 = 'SELECT Highlight { display_museum, name, highlight_description}'
result2 = client.query(query2)




# Create an empty dataframe
museum_df = pd.DataFrame(columns=['name', 'exhibits, exhibit_description'])
review_df = pd.DataFrame(columns=['museum', 'text'])
highlight_df = pd.DataFrame(columns=['display_museum', 'name', 'highlight_description'])


# rename the columns to museum so everything can be joined easily
museum_df = pd.DataFrame(result)
museum_df = museum_df.rename(columns={'name': 'museum'})
museum_df = museum_df.dropna(subset=['exhibit_description'])
museum_df['type'] = 'exhibit'
museum_df = museum_df.rename(columns={'exhibit_description': 'text'})
review_df = pd.DataFrame(result1)
review_df['type'] = 'review'
highlight_df = pd.DataFrame(result2)
highlight_df = highlight_df.rename(columns={'display_museum': 'museum'})
highlight_df = highlight_df.dropna(subset=['highlight_description'])
highlight_df['type'] = 'highlight'
museum_df = museum_df.rename(columns={'highlight_description': 'text'})
# change all pulled information into "text" column and create type column for possible tier system later - like exhibit is *3 sum of cosine similarity, highlight is *2 and review is *1.5
# reasoning is an exhibit is a collection of artwork while a highlight is one artwork and review is an opinion

#find out which names are actually in our database
candidate_names = deduplicated_df['name'].unique().tolist()
museum_database_names = museum_df['museum'].unique().tolist()

candidate_names_lower = [name.lower() for name in candidate_names]
museum_database_names_lower = [name.lower() for name in museum_database_names]

common_names = [name for name in museum_database_names_lower if any(name in candidate for candidate in candidate_names_lower)]

st.text("These are the museums currently in database")
st.text(common_names)
common_names_length = len(common_names)
st.write(f"The number of the museums returned from database is: {common_names_length}")




combined_df = pd.concat([museum_df, review_df, highlight_df], ignore_index=True)  # make on big dataframe

#st.text("This is the result of the queries to the database")
#st.dataframe(combined_df)

combined_df['museum_lower'] = combined_df['museum'].str.lower()  # this will be used with common names to create corpus
combined_df['text'] = combined_df['text'].astype(str)
# we dont need to tokenize the text if we are doing semantic search so this will be commented out
#ombined_df['analysis'] = combined_df['analysis'].str.lower() # normalize letters in the analysis
#combined_df['analysis'] = combined_df['analysis'].apply(lambda x: word_tokenize(x))  #token for embedding analysis

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

combined_df['ada_embedding'] = combined_df.text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))


st.text("This is the result of the queries to the database")
st.dataframe(combined_df)


#print(search_response)

#need to get the embedding for the interest query
target_embedding = get_embedding(interest_query, model='text-embedding-ada-002')
st.write("target_embedding:", target_embedding)


#in theory the embeddings should be done in the combined stage when we pull the data from edgedb this would make it much easier to process


# Create separate dataframes for each value in common_names
dataframes = {}
for name in common_names:
    dataframe_name = f'common_name_{name}'
    dataframes[dataframe_name] = combined_df[combined_df['museum_lower'] == name][['museum_lower', 'museum','text', 'ada_embedding']].copy()

for name, dataframe in dataframes.items():
    st.text(f'{name}:')
    st.dataframe(dataframe)


# now we have to compare the interest embedding to those in each dataframe

# Calculate cosine similarity for each dataframe
cosine_similarities = {}
for name, dataframe in dataframes.items():
    embeddings = np.array(dataframe['ada_embedding'].tolist())
    cosine_similarities[name] = np.max(cosine_similarity(embeddings, [target_embedding]))

# Find the three highest cosine similarity scores
top_three = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)[:3]

'''
# Top three museum recommendations including the cosine scores
'''

for name, similarity_score in top_three:
    st.text(f"Dataframe: {name}")
    st.text(f"Cosine Similarity Score: {similarity_score}")

# final thoughts from Howard Chang

# it should be noted that this application has a serious flaw- the ada_embeddings should be calculated asynchronously and uploaded into edgedb
# this would make the application run much faster and prevent the embeddings from being recaluclated each time the program is run
# each run of the program requires ada_embedding to be recalculated when only the target sentence inputted by the user should be recalculated
# other factors that should be addressed are adding more msueums into the database - the use of a webcrawler should be utilized

# Finally the professor in charge of this course asked if we believe anyone would pay for our databases.... I would have to say no.
# The reasoning being that all the projects essentially gathered information that was publicly available. 
# I would have said yes if the information was deeply analyzed for hidden data points that would further recommendations.
# For example if we were to identify relationships between artists, artworks, collections that could generate more accurate recommendations.
# Then this information would be worthy of being sold.

# I also believe a tier system should be used in the cosine similarity. The reasoning is that a exhibit (a collection of artowrks) is more important that a single highlight.
# and both exhibits and highlights are likely to be more important that the review of one singular person (opinions- objective vs subjective).
