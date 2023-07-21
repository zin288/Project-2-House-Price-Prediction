# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

# Import ML libraries
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
import pickle

from pathlib import Path

global df, df_filtered, df_filtered_num, df_filtered_cat, user_fr_dict

## Set Page configuration ------------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title='Predict Housing Prices', page_icon='🏠', layout='wide', initial_sidebar_state='expanded')
# Set title of the app
st.title('🏠 :blue[Your House, Your Future]🔮')
st.markdown("***Make your real estate plans with technology of the future***")


## Preparing data ---------------------------------------------------------------------------------------------------------------------------------

# Using .cache_data to reduce lag
@st.cache_data
def get_data(filename):

    df =pd.read_csv(filename)

    df_filtered = df[[  # Categorical data:
                        'town', 'storey_range', 'full_flat_type', 'pri_sch_name',  
                        # Numerical data:
                        'floor_area_sqm', 'lease_commence_date', 'mall_nearest_distance', 'hawker_nearest_distance', 'mrt_nearest_distance', 
                        'pri_sch_nearest_distance', 'sec_sch_nearest_dist', 'resale_price']]
   # Model's Numerical data only
    df_filtered_num = df[[  'floor_area_sqm', 'lease_commence_date', 'mrt_nearest_distance', 'hawker_nearest_distance',
                            'mall_nearest_distance', 'pri_sch_nearest_distance', 'sec_sch_nearest_dist', 'resale_price']]

    # Model's Categorical data only 
    df_filtered_cat = df[['town', 'storey_range', 'full_flat_type', 'pri_sch_name']]

    # user_fr_dict will store the caterogrical values in a user-friendly form,
    # by removing '_' and capitalising first letter of each word
    user_fr_dict = {}

    # Iterate over each column in df_filtered_cat, get the unique values, and add to dictionary
    for col in df_filtered_cat.columns:
        unique_values = df_filtered_cat[col].unique()
        transformed_unique_values = [value.replace('_', ' ').title() for value in unique_values]
        user_fr_dict[col] = transformed_unique_values

    return df, df_filtered, df_filtered_num, df_filtered_cat, user_fr_dict

df, df_filtered, df_filtered_num, df_filtered_cat, user_fr_dict = get_data(Path(__file__).parent /'housing_df.csv')

## Feature 1: Price Predictor --------------------------------------------------------------------------------------------------------------------

# Taking user input for price prediction
def get_predictors():
    st.sidebar.header('Select housing attributes')

    # Drop down bars are for categorical features, and sliders are for numerical features

    # Town 
    town_user = st.sidebar.selectbox('Town',(sorted(user_fr_dict['town'])))
    # # Convert the chosen word back to its original dataset form
    town = town_user.replace(' ', '_').lower()

    # Floor Range  
    range_3 = ['10_to_12', '07_to_09', '13_to_15', '01_to_03', '28_to_30', '19_to_21', '04_to_06', '16_to_18', '22_to_24',  '34_to_36', '25_to_27', 
   '37_to_39', '31_to_33', '43_to_45', '40_to_42', '49_to_51', '46_to_48']
    floor_range_user = st.sidebar.selectbox('Storey Range', (sorted(range_3)))
    floor_range = floor_range_user.replace(' ', '_').lower()
    # Note that floor_range that are grouped into 5 floors (which overlaps with ranges above, and takes up 2.97% of the entire dataset) are not provided as options, to prevent confusion.

    # Full Flat Type 
    flat_user = st.sidebar.selectbox('Full Flat Type',(sorted(user_fr_dict['full_flat_type'])))
    full_flat_type = flat_user.replace(' ', '_').lower()

    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', int(df_filtered['floor_area_sqm'].min()), int(df_filtered['floor_area_sqm'].max()),int(df_filtered['floor_area_sqm'].min()))
    lease_commence_date = st.sidebar.slider('Lease Commencement Date', int(df_filtered['lease_commence_date'].min()), int(df_filtered['lease_commence_date'].max()),int(df_filtered['lease_commence_date'].min()))

    # Nearest Primary School
    pri_sch_user = st.sidebar.selectbox('Nearest Primary School',(sorted(user_fr_dict['pri_sch_name'])))
    primary_school = pri_sch_user.replace(' ', '_').lower()

    st.sidebar.subheader('Distance of the nearest...')
    st.sidebar.markdown('<div style="text-align: right;">(Ref: 500 meters ≈ 6 min)</div>', unsafe_allow_html=True)

    pri_sch_nearest_distance = st.sidebar.slider('Primary School',  float(df_filtered['pri_sch_nearest_distance'].min()), float(df_filtered['pri_sch_nearest_distance'].max()), float(df_filtered['pri_sch_nearest_distance'].min()))
    sec_sch_nearest_dist = st.sidebar.slider('Secondary School',  float(df_filtered['sec_sch_nearest_dist'].min()), float(df_filtered['sec_sch_nearest_dist'].max()), float(df_filtered['sec_sch_nearest_dist'].min()))
    mall_nearest_distance = st.sidebar.slider('Mall', float(df_filtered['mall_nearest_distance'].min()), float(df_filtered['mall_nearest_distance'].max()),float(df_filtered['mall_nearest_distance'].min()))
    hawker_nearest_distance = st.sidebar.slider('Hawker',  float(df_filtered['mall_nearest_distance'].min()), float(df_filtered['mall_nearest_distance'].max()),float(df_filtered['mall_nearest_distance'].min()))
    mrt_nearest_distance = st.sidebar.slider('MRT',float(df_filtered['mrt_nearest_distance'].min()), float(df_filtered['mrt_nearest_distance'].max()), float(df_filtered['mrt_nearest_distance'].min()))

    return town, floor_range, full_flat_type, floor_area_sqm, lease_commence_date, primary_school, pri_sch_nearest_distance, sec_sch_nearest_dist, mall_nearest_distance, hawker_nearest_distance, mrt_nearest_distance

town, floor_range, full_flat_type, floor_area_sqm, lease_commence_date, primary_school, pri_sch_nearest_distance, sec_sch_nearest_dist, mall_nearest_distance, hawker_nearest_distance, mrt_nearest_distance = get_predictors()

# Model and Prediction
def price_predictor():
    # List of categorical features to encode and numerical features to select
    cat_features = df_filtered_cat.columns
    num_features = df_filtered_num.columns[df_filtered_num.columns != 'resale_price'].tolist()

    # Initialise One Hot Encoder & Column Transformer
    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    transformer_m1 = ColumnTransformer([("enc", onehotencoder, cat_features)], remainder = "passthrough")

    # Fit the categorical feature names of the training dataset to the transformer
    enc_data_m1 = transformer_m1.fit(df_filtered[cat_features])

    # Function to perform the one hot encoding based on the selected categorical features from training data
    def get_one_hot_encoded_m1 (data):
        
        # Fill the train values
        enc_df = pd.DataFrame(transformer_m1.transform(data[cat_features]).toarray())

        # Get the column Names
        enc_df.columns = transformer_m1.get_feature_names_out(cat_features)

        # merge the encoded values with the numerical features
        merged_df = data[num_features].join(enc_df)
        
        return merged_df

    # Retreiving LR model file
    try:
        with open(Path(__file__).parent /'model1.sav', 'rb') as file:
            model1 = pickle.load(file)
    except FileNotFoundError:
        st.error("Failed to load the model file. Make sure it exists in the current directory.")

    # Applying OHE onto user input (predictors)
    user_input = [[ town, floor_range, full_flat_type, primary_school, floor_area_sqm, 
                    lease_commence_date, mall_nearest_distance, hawker_nearest_distance,
                    mrt_nearest_distance, pri_sch_nearest_distance, sec_sch_nearest_dist]]
    columns = [col for col in df_filtered.columns if col != 'resale_price']
    df_for_ohe = pd.DataFrame(user_input, columns=columns)
    user_input_ohe = get_one_hot_encoded_m1(df_for_ohe)

    # Generate prediction based on user selected attributes
    y_pred = model1.predict(user_input_ohe)
    formatted_pred = 'SGD ${:,.0f}'.format(y_pred[0])

    df_compare = df[(df['full_flat_type'] == full_flat_type)]
    df_compare = df_compare['resale_price'].mean()

    # Print predicted housing
    st.title("Resale Housing Price Predictor")

    # Displaying metric 
    delta = y_pred[0]- df_compare
    diff = 'higher' if delta > 0 else 'lower'
    st.metric(label="House Value", value=formatted_pred, delta=f"{'{:,.0f}'.format(int(delta))} SGD, compared to units of same housing type", 
        delta_color='normal', help='Accuracy: 89%')

    # Adding resale_price to dataframe for the creation of table feature
    df_for_ohe['resale_price'] = int(y_pred[0])
    return df_for_ohe

data_info = price_predictor()

## Feature 2: Predicted Price Comparison ---------------------------------------------------------------------------------------------------------

def prediction_comparison():

    ## Feature 2A: Creating Comparison Table by concatenating new prediction data with the previous one:------------------------------------------

    # Initialize table if it is not in session state
    if 'table' not in st.session_state:
        st.session_state.table = pd.DataFrame(columns=df_filtered.columns)

    # Multiple cols created here so that the buttons can be closer next to each other
    col1, col2, col3, col4, col5 = st.columns(5, gap='small')

    # A new row of housing data will get added to the table when this button is pressed
    if col1.button("Save Data for Housing Comparison"):
        # Concatenate the new DataFrame to the existing value_df
        st.session_state.table = pd.concat([st.session_state.table, data_info], ignore_index=True)

    # Only show the table when at least 1 set of housing data is saved
    if st.session_state.table.shape[0]>0:
        st.dataframe(st.session_state.table)

        # All data is being reset when this button is pressed
        if col2.button("Reset Table"):
            st.session_state.table = pd.DataFrame(columns=df_filtered.columns)

    ## Feature 2B: Plotting visualisations for better comparison of properties ---------------------------------------------------------------

        # Split page into 2 columns - 1 visualisation in each column
        col1, col2 = st.columns(2)
        df_compare = st.session_state.table

        ##################################################################################################################################
        # Approach 1: Plot resale_price against different attributes.
        # !! Issue with Approach 1: when 2 sets of data have the same categorical value, it results in a stacked bar plot, instead of side by side. 
        # resulting in a confusing visulaisation

        # col1.subheader("Comparing housing attributes against predicted price")
        
        # # Allow the user to select columns and values
        # selected_column = col1.selectbox("", df_compare.columns)

        # if selected_column in df_filtered_cat:
        #     fig = px.bar(df_compare, x=selected_column, y="resale_price", opacity=0.9, color=selected_column, text=df_compare.index)

        # else:
        #     fig = px.scatter(df_compare, x=selected_column, y="resale_price", opacity=0.9, color=selected_column, text=df_compare.index, color_continuous_scale="YlOrRd")
        #     # Customize the dot size
        #     fig.update_traces(marker=dict(size=30), textfont=dict(color='black'))

        ##################################################################################################################################

        # Approach 2: Plot attributes against property index values.
        # # !! Issue with Appraoch 2: The circle size is supposed to reflect price value, but it is not reflecting that properly

        col1.subheader("Comparing housing attributes of selected properties")

        # Allow the user to select columns and values
        selected_column = col1.selectbox("", [col for col in df_compare.columns if col != 'resale_price'])

        # Set the index values as a separate column, to change what the users see as the x-axis title
        df_compare['property_index'] = df_compare.index

        if selected_column in df_filtered_cat:
            # Changing the resale_price values into '$123,456' form
            df_compare['predicted_price'] = df_compare['resale_price'].apply(lambda x: "${:,.0f}".format(x))
            fig = px.scatter(df_compare, x='property_index', y=selected_column, opacity=0.9, color=selected_column, text='predicted_price', color_continuous_scale="YlOrRd")
            
            # The following section changes the size of the scatter points depending on value of house
            try:
                # Calculate the normalized size based on the resale_price column. 
                normalized_size = ((df_compare['resale_price'] - df_compare['resale_price'].min())  / (df_compare['resale_price'].max() - df_compare['resale_price'].min()))* 100 + 100
                # Set the size of the dots based on the normalized resale_price values
                dot_size = normalized_size.values.tolist()  # Adjust the scaling factor as per your preference
            except:
                # Program goes into 'except' if denominator is 0 (when only one house is selected). In this case, set size to be 100.
                dot_size = 100
            # Customize the dot size
            fig.update_traces(marker=dict(size=dot_size), textfont=dict(color='black'))
            # Set the x-axis to display only integer values
            fig.update_xaxes(tickmode="array", tickvals=df_compare.index, ticktext=df_compare.index.astype(int))
        else:
            # Create the bar plot
            fig = px.bar(df_compare, x='property_index', y=selected_column, opacity=0.9)
            # Set the orientation of the bars to 'v' (vertical)
            fig.update_traces(orientation='v')
            # Set the x-axis to display only integer values
            fig.update_xaxes(tickmode='linear')

        ##################################################################################################################################
            
        # Plotting the chart
        col1.plotly_chart(fig)

        col2.subheader("Comparing between housing attributes")

        selected_column2 = col2.selectbox("Select attribute 1", df_compare.columns)
        selected_column3 = col2.selectbox("Select attribute 2", df_compare.columns)

        if selected_column2 in df_filtered_cat:
            fig = px.bar(df_compare, x=selected_column2, y=selected_column3, opacity=0.9, color=selected_column, text=df_compare.index)
        if selected_column2 in df_filtered_num:
            fig = px.scatter(df_compare, x=selected_column2, y=selected_column3, opacity=0.9, color=selected_column, text=df_compare.index, color_continuous_scale="YlOrRd")
            # Customize the dot size
            fig.update_traces(marker=dict(size=30), textfont=dict(color='black'))
        col2.plotly_chart(fig)

prediction_comparison()