# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go


from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from patsy import dmatrices, dmatrix
from yellowbrick.regressor import prediction_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest

import pickle


## Set Page configuration

st.set_page_config(page_title='Predict Housing Prices', page_icon='🏠', layout='wide', initial_sidebar_state='expanded')
# Set title of the app
st.title('🏠 :blue[Housing Price Predictor]🔮')
st.markdown("***Your House, Your Future: Making Informed Real Estate Choices***")


## Preparing data

# Using .cache_data so that it does not take a long time to reload
@st.cache_data
def get_data(filename):
    df = pd.read_csv(filename)
    return df

df = get_data('../datasets/housing_df.csv')

# Data for model 1
df_filtered = df[[   # Categorical data
            'town', 'storey_range', 'full_flat_type', 'pri_sch_name',  
            # Numerical data
            'floor_area_sqm', 'lease_commence_date', 'mall_nearest_distance', 'hawker_nearest_distance', 'mrt_nearest_distance', 
            'pri_sch_nearest_distance', 'sec_sch_nearest_dist', 'resale_price']]

# Numerical data only
df_filtered_num = df[['floor_area_sqm', 'lease_commence_date', 'mall_nearest_distance', 'hawker_nearest_distance', 'mrt_nearest_distance', 
                        'pri_sch_nearest_distance', 'sec_sch_nearest_dist', 'resale_price']]
# Numerical data only
df_filtered_cat = df[['town', 'storey_range', 'full_flat_type', 'pri_sch_name']]


## Feature 1: Price Predictor

# Set input widgets
st.sidebar.subheader('Select housing attributes')

# Drop down bars are for cateogrical features, and sliders are for numerical features

# Town : Converting options into user friendly forms
town_list_before = ['kallang/whampoa', 'bishan', 'bukit_batok', 'yishun', 'geylang',
       'hougang', 'bedok', 'sengkang', 'tampines', 'serangoon',
       'bukit_merah', 'bukit_panjang', 'woodlands', 'jurong_west',
       'toa_payoh', 'choa_chu_kang', 'sembawang', 'ang_mo_kio',
       'pasir_ris', 'clementi', 'punggol', 'jurong_east', 'central_area',
       'queenstown', 'bukit_timah', 'marine_parade']
# Sort in alphabetical order, replace '_' with spaces, and capitalise first letter of each word
town_list_after = sorted([town.replace('_', ' ').title() for town in town_list_before])
# Create drop down widget for town selection
town_user = st.sidebar.selectbox('Town',(town_list_after))
# Convert the chosen word back to its original dataset form
town = town_user.replace(' ', '_').lower()

# Floor Range
floor_range = st.sidebar.selectbox('Storey Range', ('01_to_03', '04_to_06', '07_to_09', '10_to_12', '13_to_15', '16_to_18', 
                                                    '19_to_21', '22_to_24', '25_to_27', '28_to_30', '31_to_33',
                                                    '34_to_36', '37_to_39', '40_to_42', '43_to_45', '46_to_48', '49_to_51'))
# Note that floor_range that are grouped into 5 floors (which overlaps with ranges above, ad takes up 2.97% of the entire dataset)
# are not provided as options, to prevent confusion.


# Full Flat Type: same operations as Town Widget
flat_list_before = ['4_room_model_a', '5_room_improved', 'executive_apartment',
        '4_room_simplified', '3_room_improved', '3_room_new_generation',
       '4_room_premium_apartment', '3_room_model_a',
       '5_room_premium_apartment', '4_room_model_a2', '5_room_model_a',
       'executive_maisonette', '4_room_new_generation',
       'executive_premium_apartment', '4_room_improved',
       '2_room_standard', '3_room_standard', '3_room_simplified',
       '2_room_improved', '5_room_standard', '3_room_premium_apartment',
       '2_room_model_a', '4_room_dbss', '3_room_terrace', '3_room_dbss',
       '5_room_dbss', '5_room_model_a-maisonette', '4_room_type_s1',
       '1_room_improved', '4_room_premium_apartment_loft',
       'executive_adjoined_flat', '4_room_terrace',
       'multi-generation_multi_generation', '4_room_standard',
       '5_room_type_s2', '5_room_adjoined_flat',
       '5_room_premium_apartment_loft', '4_room_adjoined_flat',
       'executive_premium_maisonette', '2_room_premium_apartment',
       '5_room_improved-maisonette', '2_room_2-room', '2_room_dbss']
flat_list_after = sorted([flat.replace('_', ' ').title() for flat in flat_list_before])
flat_user = st.sidebar.selectbox('Full Flat Type',(flat_list_after))
full_flat_type = flat_user.replace(' ', '_').lower()

floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', df_filtered['floor_area_sqm'].min(), df_filtered['floor_area_sqm'].max(),df_filtered['floor_area_sqm'].min())
lease_commence_date = st.sidebar.slider('Lease Commencement Date', df_filtered['lease_commence_date'].min(), df_filtered['lease_commence_date'].max(),df_filtered['lease_commence_date'].min())
mall_nearest_distance = st.sidebar.slider('Distance of the nearest mall', df_filtered['mall_nearest_distance'].min(), df_filtered['mall_nearest_distance'].max(),df_filtered['mall_nearest_distance'].min())
hawker_nearest_distance = st.sidebar.slider('Distance of the nearest Hawker',  df_filtered['mall_nearest_distance'].min(), df_filtered['mall_nearest_distance'].max(),df_filtered['mall_nearest_distance'].min())
mrt_nearest_distance = st.sidebar.slider('Distance of the nearest MRT',df_filtered['mrt_nearest_distance'].min(), df_filtered['mrt_nearest_distance'].max(), df_filtered['mrt_nearest_distance'].min())

# Nearest Primary School: same operations as Town Widget
primary_school_before = ['geylang_methodist_school',
       'kuo_chuan_presbyterian_primary_school', 'keming_primary_school',
       'catholic_high_school', 'naval_base_primary_school',
       "saint_margaret's_primary_school", 'xinmin_primary_school',
       'damai_primary_school', 'ai_tong_school',
       'anchor_green_primary_school', 'north_vista_primary_school',
       'tampines_north_primary_school', 'rosyth_school',
       'alexandra_primary_school', 'beacon_primary_school',
       'zhenghua_primary_school', 'fuchun_primary_school',
       'guangyang_primary_school', 'woodlands_ring_primary_school',
       'fernvale_primary_school', 'palm_view_primary_school',
       'greenwood_primary_school', 'frontier_primary_school',
       'lakeside_primary_school', 'chua_chu_kang_primary_school',
       'wellington_primary_school', 'hong_wen_school',
       'compassvale_primary_school', 'kheng_cheng_school',
       'bendemeer_primary_school', 'unity_primary_school',
       'ang_mo_kio_primary_school', 'marsiling_primary_school',
       'bukit_view_primary_school', 'blangah_rise_primary_school',
       'sembawang_primary_school', 'admiralty_primary_school',
       'yumin_primary_school', "saint_andrew's_junior_school",
       'south_view_primary_school', 'peiying_primary_school',
       "saint_hilda's_primary_school", 'eunos_primary_school',
       'anderson_primary_school', 'teck_ghee_primary_school',
       'white_sands_primary_school', 'farrer_park_primary_school',
       'lianhua_primary_school', 'qifa_primary_school',
       'northland_primary_school', 'horizon_primary_school',
       'jing_shan_primary_school', 'yuhua_primary_school',
       'bedok_green_primary_school', 'pei_chun_public_school',
       'canossa_catholic_primary_school', 'zhangde_primary_school',
       'evergreen_primary_school', 'fern_green_primary_school',
       'west_view_primary_school', 'first_toa_payoh_primary_school',
       'telok_kurau_primary_school', "saint_anthony's_primary_school",
       'elias_park_primary_school', 'fuhua_primary_school',
       'stamford_primary_school', 'dazhong_primary_school',
       'north_spring_primary_school', "holy_innocents'_primary_school",
       'tampines_primary_school', 'greenridge_primary_school',
       'henry_park_primary_school', 'pei_tong_primary_school',
       'hougang_primary_school', 'bukit_panjang_primary_school',
       'woodgrove_primary_school', 'mayflower_primary_school',
       'fengshan_primary_school', 'pioneer_primary_school',
       'punggol_green_primary_school', 'chongfu_school',
       'townsville_primary_school', 'xishan_primary_school',
       'punggol_primary_school', 'riverside_primary_school',
       'si_ling_primary_school', 'punggol_view_primary_school',
       'rivervale_primary_school', 'west_grove_primary_school',
       'changkat_primary_school', 'teck_whye_primary_school',
       'corporation_primary_school', 'ahmad_ibrahim_primary_school',
       'angsana_primary_school', 'woodlands_primary_school',
       'kong_hwa_school', 'nan_chiau_primary_school',
       'boon_lay_garden_primary_school', 'gan_eng_seng_primary_school',
       'westwood_primary_school', 'jiemin_primary_school',
       'pei_hwa_presbyterian_primary_school', 'poi_ching_school',
       'oasis_primary_school', 'kranji_primary_school',
       'new_town_primary_school', 'chij_our_lady_of_good_counsel',
       'rulang_primary_school', 'yu_neng_primary_school',
       'north_view_primary_school', 'northoaks_primary_school',
       'east_spring_primary_school', 'yangzheng_primary_school',
       'canberra_primary_school', 'mee_toh_school',
       'greendale_primary_school', 'jurong_west_primary_school',
       'springdale_primary_school', 'maha_bodhi_school',
       'juying_primary_school', 'concord_primary_school',
       'innova_primary_school', 'junyuan_primary_school',
       'chij_our_lady_of_the_nativity', 'qihua_primary_school',
       'radin_mas_primary_school', 'nan_hua_primary_school',
       'meridian_primary_school', 'de_la_salle_school',
       'fairfield_methodist_school', 'seng_kang_primary_school',
       "paya_lebar_methodist_girls'_school", 'shuqun_primary_school',
       'cantonment_primary_school', 'sengkang_green_primary_school',
       'clementi_primary_school', "chij_saint_nicholas_girls'_school",
       'yishun_primary_school', 'xinghua_primary_school',
       'huamin_primary_school', 'edgefield_primary_school',
       'gongshang_primary_school', 'queenstown_primary_school',
       'chongzheng_primary_school', 'park_view_primary_school',
       'princess_elizabeth_primary_school', 'casuarina_primary_school',
       'yio_chu_kang_primary_school', 'opera_estate_primary_school',
       'west_spring_primary_school', 'montfort_junior_school',
       'nanyang_primary_school', 'red_swastika_school',
       "saint_anthony's_canossian_primary_school",
       'zhonghua_primary_school', "saint_stephen's_school",
       'waterway_primary_school', 'chij_primary',
       'punggol_cove_primary_school', 'valour_primary_school',
       'temasek_primary_school', 'yew_tee_primary_school',
       'pasir_ris_primary_school', 'chij', 'cedar_primary_school',
       'endeavour_primary_school', "haig_girls'_school",
       'jurong_primary_school', 'maris_stella_high_school',
       'tao_nan_school', 'xingnan_primary_school',
       'ngee_ann_primary_school', "saint_joseph's_institution_junior",
       'river_valley_primary_school', 'chij_our_lady_queen_of_peace',
       'marymount_convent_school', "saint_gabriel's_primary_school"]
primary_school_after = sorted([school.replace('_', ' ').title() for school in primary_school_before])
pri_sch_user = st.sidebar.selectbox('Nearest Primary School',(primary_school_after))
primary_school = pri_sch_user.replace(' ', '_').lower()

pri_sch_nearest_distance = st.sidebar.slider('Distance of the nearest Primary School',  df_filtered['pri_sch_nearest_distance'].min(), df_filtered['pri_sch_nearest_distance'].max(), df_filtered['pri_sch_nearest_distance'].min())
sec_sch_nearest_dist = st.sidebar.slider('Distance of the nearest Secondary School',  df_filtered['sec_sch_nearest_dist'].min(), df_filtered['sec_sch_nearest_dist'].max(), df_filtered['sec_sch_nearest_dist'].min())


# Model and Prediction

# List of categorical features to encode and numerical features to select
cat_name_m1 = ['town', 'storey_range','full_flat_type', 'pri_sch_name']
num_name_m1 = [
    'floor_area_sqm',
    'lease_commence_date',
    'mrt_nearest_distance',
    'hawker_nearest_distance',
    'mall_nearest_distance',
    'pri_sch_nearest_distance',
    'sec_sch_nearest_dist']

# Initialise One Hot Encoder & Column Transformer
onehotencoder = OneHotEncoder(handle_unknown='ignore')
transformer_m1 = ColumnTransformer([("enc",
                                   onehotencoder,
                                   cat_name_m1)],
                                   remainder = "passthrough")
# Fit the categorical feature names of the training dataset to the transformer
enc_data_m1 = transformer_m1.fit(df[cat_name_m1])

# Create a function to perform the one hot encoding based on the selected categorical features from training data
def get_one_hot_encoded_m1 (data):
    
    # Fill the train values
    enc_df = pd.DataFrame(transformer_m1.transform(data[cat_name_m1]).toarray())

    # Get the column Names
    enc_df.columns = transformer_m1.get_feature_names_out(cat_name_m1)

    # merge the encoded values with the numerical features
    merged_df = data[num_name_m1].join(enc_df)
    
    return merged_df


filename = '../models/model1.sav'
model1 = pickle.load(open(filename, 'rb'))

user_input = [[ town, floor_range, full_flat_type, primary_school,
    floor_area_sqm, lease_commence_date, mall_nearest_distance, hawker_nearest_distance,mrt_nearest_distance, pri_sch_nearest_distance, sec_sch_nearest_dist]]
columns = ['town', 'storey_range', 'full_flat_type', 'pri_sch_name',
           'floor_area_sqm', 'lease_commence_date', 'mrt_nearest_distance',
           'hawker_nearest_distance', 'mall_nearest_distance',
           'pri_sch_nearest_distance', 'sec_sch_nearest_dist']

df_for_ohe = pd.DataFrame(user_input, columns=columns)
user_input_ohe = get_one_hot_encoded_m1(df_for_ohe)

# Generate prediction based on user selected attributes
y_pred = model1.predict(user_input_ohe)
formatted_pred = 'SGD ${:,.0f}'.format(y_pred[0])

#df_compare = df[(df['town'] == town) & (df['full_flat_type'] == full_flat_type)]
df_compare = df[(df['full_flat_type'] == full_flat_type)]
df_compare = df_compare['resale_price'].mean()


# Print predicted housing
st.subheader('Predicted Housing Price')
# st.metric('Predicted Housing Price is :', formatted_pred, '')
# Display a metric with a delta value and color
delta = y_pred[0]- df_compare
diff = 'higher' if delta > 0 else 'lower'
# delta = formatted_pred * 1.0 - df_compare
# diff = 'higher'
st.metric(label="", value=formatted_pred, delta=f"{'{:,.0f}'.format(int(delta))} SGD, compared to units of same housing type", 
    delta_color='normal', help='Accuracy: 89%')



## Feature 2A: EDA - Numerical predictor scatter plot

st.title("Past Resale Transaction Insights")

# Allow the user to select columns and values
selected_column = st.selectbox("Select a column", df_filtered_num.columns)
# selected_value = st.number_input("Enter a value")

# Filter the DataFrame based on the user's selection
filtered_user_df = df_filtered[selected_column]

fig = px.scatter(df_filtered, x=selected_column, y="resale_price" )

st.plotly_chart(fig)


## Feature 2B: EDA - Resale prices by town Scatter plot

# Calculate average resale prices by town
st.subheader("Average Resale Prices by Town")

filtered_user_df = pd.concat([df_filtered_cat, df_filtered_num['resale_price']], axis=1)

selected_column = st.selectbox("Select a column", df_filtered_cat.columns)

# Calculate average resale prices by town
average_prices = filtered_user_df.groupby(selected_column)["resale_price"].mean().sort_values().reset_index()

fig = px.bar(average_prices, x=selected_column, y="resale_price", color=selected_column, title="Average Resale Prices by Town")
fig.update_layout(xaxis_title="Town", yaxis_title="Average Resale Price")

st.plotly_chart(fig)




## Feature 3C: Map

# Get unique town values from the DataFrame
towns = df['town'].unique().tolist()

st.subheader("A closer look at each transaction")

# Create a selection widget for the town
selected_town = st.selectbox("Select a town", towns)

# Filter the DataFrame based on the selected town
selected_df = df[df['town'] == selected_town]

# Get the latitude and longitude coordinates of the selected town
selected_lat = selected_df['latitude'].values[0]
selected_lon = selected_df['longitude'].values[0]

# Create a Folium map centered on the selected town
m = folium.Map(location=[selected_lat, selected_lon], zoom_start=14, tiles='CartoDB positron')

# Iterate over each row in the selected DataFrame
for index, row in selected_df.iterrows():
    # Extract the latitude and longitude values
    lat = row['latitude']
    lon = row['longitude']

    # Extract additional information
    town_name = row['town'].replace("_", " ").capitalize()
    address = "Blk " + row['block'].replace("_", " ").capitalize() +' ' + row['street_name'].replace("_", " ").capitalize()
    price = row['resale_price']
    info = f"Town: {town_name}<br>Address: {address}<br>'Resale Price: ${price}"

    # Create a marker at the latitude and longitude coordinates
    marker = folium.Marker([lat, lon], popup=folium.Popup(info, max_width=250))
    marker.add_to(m)

# Display the map using Streamlit
st.markdown("**Click on the marker to see unit information**")
# st.markdown("Dots represent transactions")
folium_static(m)



df_floors = df[df['storey_range'].isin(['01_to_03', '04_to_06', '07_to_09', '10_to_12', '13_to_15', '16_to_18', '19_to_21', '22_to_24', '25_to_27', '28_to_30', '31_to_33',
                                                    '34_to_36', '37_to_39', '40_to_42', '43_to_45', '46_to_48', '49_to_51'])]

df_floors['storey_range'] = df_floors['storey_range'].str.replace('_', ' ')

# Extract the first 2 digits from the storey_range column
df_floors['range_digits'] = df_floors['storey_range'].str[:2]

# Sort the dataframe by range_digits in ascending order
df_floors.sort_values('range_digits', inplace=True)


# Create a boxplot using Plotly
fig = px.box(df_floors, x='storey_range', y='resale_price', color='storey_range',
             color_discrete_sequence=px.colors.sequential.Blues)
fig.update_xaxes(tickangle=45)
# Customize the layout
fig.update_layout(
    title='Boxplot of Resale Price by Storey Range',
    xaxis_title='Storey Range',
    yaxis_title='Resale Price'
)

# Display the plot using Streamlit
st.plotly_chart(fig)




# # Create the regression plot using Plotly
# fig = px.scatter(df_floors, x='lease_commence_date', y='resale_price', trendline='ols')

# # Customize the trendline color and thickness
# fig.update_traces(selector=dict(name='trendline'), line_color='maroon', line_width=3)

# # Customize the layout
# fig.update_layout(
#     title='Flat Lease Commence Date vs Resale Price',
#     xaxis_title='Lease Commence Date',
#     yaxis_title='Resale Price (SGD)',
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig)
