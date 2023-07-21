# Import Libraries
import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path


# ## Set Page configuration -------------------------------------------------------------------------------------------------------------------------

st.title('🏠 :blue[Your House, Your Future]🔮')
st.markdown("***Make your real estate plans with technology of the future***")

st.subheader("A closer look at each past transaction")
# Embed the Tableau Dashboard
html_temp = """<div class='tableauPlaceholder' id='viz1688839675413' style='position: relative'><noscript><a href='#'><img alt='Maps ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;SG&#47;SG_HDB_Resale_Price&#47;Maps&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='views&#47;SG_HDB_Resale_Price&#47;Maps?:language=en-GB&amp;:embed=true' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;SG&#47;SG_HDB_Resale_Price&#47;Maps&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1688839675413');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
components.html(html_temp, width=1300, height=800)

# st.title('🔧 Premium content coming your way... ')

# Set title of the app
#st.title('🏠 Page 1🔮')
# st.markdown("Please support our efforts in empowering all in their real estate journey ❤️")



# Define the layout of your Streamlit app
# st.title("Resale Price Insights")
