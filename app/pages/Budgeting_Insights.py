# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# ## Set Page configuration -------------------------------------------------------------------------------------------------------------------------

st.title('🏠 :blue[Your House, Your Future]🔮')
st.markdown("***Make your real estate plans with technology of the future***")


## Preparing data -----------------------------------------------------------------------------------------------------------------------------------

# Using .cache_data so to reduce lag
@st.cache_data
def get_data(filename):
    df = pd.read_csv(filename)

    # Data needed for model 1
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


    # user_fr_dict will store the caterogrical values as a user-friendly form,
    # by removing '_' and capitalising first letter of each word
    user_fr_dict = {}

    # Iterate over each column in df_filtered_cat, get the unique values, and add to dictionary
    for col in df_filtered_cat.columns:
        unique_values = df_filtered_cat[col].unique()
        transformed_unique_values = [value.replace('_', ' ').title() for value in unique_values]
        user_fr_dict[col] = transformed_unique_values

    return df, df_filtered, df_filtered_num, df_filtered_cat, user_fr_dict

df, df_filtered, df_filtered_num, df_filtered_cat, user_fr_dict = get_data(Path(__file__).parent /'../housing_df.csv')

## Feature 2: EDA - ----------------------------------------------------------------------------------------------------------------------------------
@st.cache_data(experimental_allow_widgets=True)
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
	# filtered_user_df = df_filtered[selected_column]
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

		fig.update_layout(width=1200, height=600, margin=dict(l=200, r=0, t=50, b=50, pad=4))
	st.plotly_chart(fig)

	return budget_min, budget_max

budget_min, budget_max = show_eda()

if 'budget_min' not in st.session_state:
    st.session_state['budget_min'] = budget_min
    
if 'budget_max' not in st.session_state:
    st.session_state['budget_max'] = budget_max

## Feature 4: Other EDAs --------------------------------------------------------------------------------------------------------------------------------

st.subheader("Want more detailed analysis?")

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



# st.title('🔧 Premium content coming your way... ')

# Set title of the app
#st.title('🏠 Page 1🔮')
# st.markdown("Please support our efforts in empowering all in their real estate journey ❤️")



# Define the layout of your Streamlit app
# st.title("Resale Price Insights")
