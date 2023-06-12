# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from Resale_Price_Predictor import df
from Resale_Price_Predictor import df_filtered
from Resale_Price_Predictor import df_filtered_num
from Resale_Price_Predictor import df_filtered_cat
from Resale_Price_Predictor import user_fr_dict


# ## Set Page configuration -------------------------------------------------------------------------------------------------------------------------

st.title('🏠 :blue[Your House, Your Future]🔮')
st.markdown("***Make your real estate plans with technology of the future***")

## Feature 3: EDA - ----------------------------------------------------------------------------------------------------------------------------------

def show_eda():
	# Set budget
	st.title("Past Resale Transaction Insights")
	st.subheader("Select Budget Range")

	# After a user inputs a budget, only feature values with resale_price within this range will be shown

	# Create a range slider widget for the budget
	budget_min, budget_max = st.slider("", int(df['resale_price'].min()), int(df['resale_price'].max()), (0, int(df['resale_price'].max())))

	# Display the selected budget range
	st.write("Selected Budget Range:", budget_min, '-', budget_max)

	# Define the range of y-values to color in red
	y_range = [budget_min, budget_max] 

	# Allow the user to select columns and values
	selected_column = st.selectbox("Select an attribute", df_filtered.columns)

	# Filter the DataFrame based on the user's selection
	filtered_user_df = pd.concat([df_filtered_cat, df[[ 'mrt_name', 'sec_sch_name']]], axis=1)
	filtered_user_df = pd.concat([df_filtered_cat, df_filtered['resale_price']], axis=1)

	if selected_column not in df_filtered_cat:
		# Create a new column indicating whether each data point falls within the y-value range
		df_filtered['color'] = np.where((df_filtered['resale_price'] >= y_range[0]) &
		                                     (df_filtered['resale_price'] <= y_range[1]),
		                                     'maroon', 'blue')
		fig = px.scatter(df_filtered, x=selected_column, y="resale_price", color='color')

	else:
		# Calculate average resale prices by col
		average_prices = filtered_user_df.groupby(selected_column)["resale_price"].mean().sort_values().reset_index()
		filtered_prices = average_prices[
	    (average_prices['resale_price'] >= budget_min) &
	    (average_prices['resale_price'] <= budget_max)
	]

		fig = px.bar(filtered_prices, x=selected_column, y="resale_price", color=selected_column)

		fig.update_layout(width=1200, height=600)
	st.plotly_chart(fig)

	return budget_min, budget_max

budget_min, budget_max = show_eda()

## Feature 4: Map -----------------------------------------------------------------------------------------------------------------------------------------
def show_map():
	# Get unique town values from the DataFrame
	towns = df['town'].unique().tolist()

	st.subheader("A closer look at each transaction")
	st.markdown("Note: Red markers are within budget range")

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
	    info = f"Town: {town_name}<br>Address: {address}<br>Resale Price: ${price}"

	       # Check if the resale price is within the budget range
	    if budget_min <= price <= budget_max:
	        # Create a marker at the latitude and longitude coordinates with red color
	        marker = folium.Marker([lat, lon], popup=folium.Popup(info, max_width=250), icon=folium.Icon(color='red'))
	    else:
	        # Create a marker at the latitude and longitude coordinates with default color
	        marker = folium.Marker([lat, lon], popup=folium.Popup(info, max_width=250))

	    marker.add_to(m)

	st.markdown("**Click on the marker to see unit information**")
	folium_static(m)

show_map()

## Feature 5: Other EDAs --------------------------------------------------------------------------------------------------------------------------------

def show_boxplot():

	st.subheader("Want more detailed analysis?")

	df_floors = df[df['storey_range'].isin(['01_to_03', '04_to_06', '07_to_09', '10_to_12', '13_to_15', '16_to_18', '19_to_21', '22_to_24', '25_to_27', '28_to_30', '31_to_33',
	                                                    '34_to_36', '37_to_39', '40_to_42', '43_to_45', '46_to_48', '49_to_51'])]

	df_floors.loc[:, 'storey_range'] = df_floors['storey_range'].str.replace('_', ' ')

	# Extract the first 2 digits from the storey_range column
	df_floors.loc[:,'range_digits'] = df_floors['storey_range'].str[:2]

	# Sort the dataframe by range_digits in ascending order
	df_floors = df_floors.sort_values('range_digits').copy()

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

	st.plotly_chart(fig)

show_boxplot()

st.title('🔧 Premium content coming your way... ')

# Set title of the app
st.markdown("Please support our efforts in empowering all in their real estate journey ❤️")