import streamlit as st
import joblib
import json
import os
import numpy as np
import pandas as pd

# Path Setup:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
JSON_PATH = os.path.join(BASE_DIR, "..", "data", "interim", "category_mappings.json")

# 1. Load the models
@st.cache_resource # Keep models in memory for speed
def load_assets():
    model = joblib.load(os.path.join(MODEL_DIR, "supervised_model_final_boost.pkl")) # Champio supervised model
    kmeans = joblib.load(os.path.join(MODEL_DIR, "unsupervised_kmeans_final.pkl")) # Unsupervised model
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_WITHOUT_outliers.pkl")) # Scaler for preprocessing

    with open(JSON_PATH, "r") as f:
        mappings = json.load(f) # Load category mappings for any necessary encoding
    # Helper to convert JSON list to {Name: Index} dictionary
    def list_to_dict(name_list):
        return {name: i for i, name in enumerate(name_list)}
    return model, kmeans, scaler, mappings, list_to_dict

model, kmeans, scaler, mappings, list_to_dict = load_assets()

# Human-Readable Mappings
type_map = list_to_dict(mappings["Type"])
shipping_mode_map = list_to_dict(mappings["Shipping_Mode"])
shipping_day_map = list_to_dict(mappings["shipping_day"])
city_map = list_to_dict(mappings["Order_City"]) # Dynamic City Mapping
cat_map = list_to_dict(mappings["Category_Name"]) # Dynamic Category Mapping

# UI header
st.title("DataCo Supply Chain Risk Predictor")
st.markdown("Predict delivery delays and Identify logistic risk clusters in real-time.")

# 2. Side bar for user input (The Predictor)

st.sidebar.header("Input Order Details")

# Categorical inputs
st.sidebar.header("Step 1: Order Logistics")
selected_city = st.sidebar.selectbox("Destination City", options= sorted(list(city_map.keys())))
selected_mode = st.sidebar.selectbox("Shipping Mode", options= list(shipping_mode_map.keys()))
selected_day = st.sidebar.selectbox("Shipment Day", options= list(shipping_day_map.keys()))

st.sidebar.header("Step 2: Product & Payment")
selected_cat = st.sidebar.selectbox("Category", options=sorted(list(cat_map.keys())))
selected_type = st.sidebar.selectbox("Payment Type", options=list(type_map.keys()))

# Numerical inputs
HISTORICAL_AVG_RISK = 0.5473
st.sidebar.header("Step 3: Constraints")
scheduled_days = st.sidebar.slider("Scheduled Days", 1, 6, 3)
price = st.sidebar.number_input("Unit Price ($)", value=150.0)
benefit = st.sidebar.number_input("Expected Benefit ($)", value=50.0)

# 3. Execution Logic
# Prepare data for model (Ensure columns match df_final/X_train order)
# Define the complete list of predictors in the exact training order
predictors = [
    'Days_for_shipment_scheduled', 'Benefit_per_order', 'Order_Item_Discount', 
    'Order_Item_Discount_Rate', 'Order_Item_Profit_Ratio', 'Order_Item_Quantity', 
    'Type_num', 'Category_Name_num', 'Customer_City_num', 'Customer_Country_num', 
    'Customer_Segment_num', 'Customer_State_num', 'Department_Name_num', 
    'Order_City_num', 'Order_Country_num', 'Order_State_num', 'Order_Status_num', 
    'Shipping_Mode_num', 'Customer_Zipcode_num', 'shipping_day_num', 
    'shipping_month_num', 'Price_Per_Unit', 'Logistics_Corridor_ID'
]

# Create the input row. 
# We use Sidebar Variables for user choices and THE MEDIANS for the rest.
input_values = [
    scheduled_days,              # User Input
    benefit,                     # User Input
    14.0,                        # Median Order_Item_Discount
    0.1,                         # Median Order_Item_Discount_Rate
    0.27,                        # Median Order_Item_Profit_Ratio
    1.0,                         # Median Order_Item_Quantity
    type_map[selected_type],     # User Input
    cat_map[selected_cat],       # User Input
    61.0,                        # Median Customer_City_num
    1.0,                         # Median Customer_Country_num
    0.0,                         # Median Customer_Segment_num
    1.0,                         # Median Customer_State_num
    3.0,                         # Median Department_Name_num
    city_map[selected_city],     # User Input
    18.0,                        # Median Order_Country_num
    119.0,                       # Median Order_State_num
    2.0,                         # Median Order_Status_num
    shipping_mode_map[selected_mode], # User Input
    176.0,                       # Median Customer_Zipcode_num
    shipping_day_map[selected_day],   # User Input
    6.0,                         # Median shipping_month_num
    price,                       # User Input
    310.0                        # Median Logistics_Corridor_ID
]

input_data = pd.DataFrame([input_values], columns=predictors)

# Features from the Streamlit UI inputs
input_data['Days_for_shipment_scheduled'] = scheduled_days
input_data['Benefit_per_order'] = benefit
input_data['Price_Per_Unit'] = price
input_data['Shipping_Mode_num'] = shipping_mode_map[selected_mode]
input_data['shipping_day_num'] = shipping_day_map[selected_day]
input_data['Type_num'] = type_map[selected_type]
input_data['Order_City_num'] = city_map[selected_city]
input_data['Category_Name_num'] = cat_map[selected_cat]

# Define the Human-Readable Cluster Labels
cluster_status = {
    0: "🟡 Moderate Risk (Standard)",
    1: "🟢 Low Risk (Optimal)",
    2: "🔴 Critical Risk (Impossible Schedule)"
}

# Scaling for unsupervised model
input_scaled = scaler.transform(input_data)

if st.button("Analyze Order Risk"):
    # Supervised prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] # Probability of delay

    # Unsupervised cluster assigment
    cluster = kmeans.predict(input_scaled)[0]
    readable_cluster = cluster_status.get(cluster, "Unknown Cluster")

    # 4. Interactive UI display
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        # Calculate how much higher/lower this order is compared to the 54.73% baseline
        risk_diff = (prob - HISTORICAL_AVG_RISK) * 100

        st.metric(label= "Late Risk Probability", value= f"{prob * 100:.1f}%", delta= f"{risk_diff:.1f}% vs. History", delta_color= "inverse") # Red if risk_diff is positive, Green if negative
    with col2:
        if prediction == 1:
            st.error("Status: LATE EXPECTED")
        else:
            st.success("Status: ON TIME")
    
    with col3:
        st.metric("Logistic Profile", readable_cluster)
        if cluster == 2:
            st.warning("**Strategic Insight**: This order is being promised too fast for our current logistics capacity. Recommend increasing the scheduled days to at least 3.")
        elif cluster == 1:
            st.info("**Optimization Tip**: This profile is highly efficient. Continue using these parameters for this route.")

    # 5. Interactive Map for company visibility
    st.subheader("Order Location (Puerto Rico Hub)")
    # We will generate dummy coordinates based on City IDs for visual impact
    map_data = pd.DataFrame({"lat": [18.4655], # This is Puerto Rico's Center
                             "lon": [-66.1057]})
    st.map(map_data)


# 6. Dynamic Business Recommendation
    st.divider()
    st.subheader("Strategic Recommendation")

    if cluster == 2:
        st.error("Action Required: Reschedule Order")
        st.write(f"The current promise of **{scheduled_days} day(s)** is physically impossible for our current logistics to {selected_city}.")
        
        # Calculate the 'Safe' target
        suggested_days = 4 # Based on Cluster 1 average
        additional_days = suggested_days - scheduled_days
        
        st.info(f"**To move this to 'Low Risk' (Cluster 1):** Increase the 'Scheduled Days' to **{suggested_days}**. "
                f"This adds {additional_days} day(s) to the customer promise but ensures an 85%+ on-time delivery rate.")
    
    elif cluster == 0:
        st.warning("Action Recommended: Review Buffer")
        st.write("This order is in the 'Moderate' zone. Adding **1 extra day** to the schedule would likely shift this into the 'Low Risk' green zone.")
        
    else:
        st.success("No Action Needed")
        st.write("The current scheduling window is optimal. This order is highly likely to meet its deadline.")




