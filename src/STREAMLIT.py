import streamlit as st
import joblib
import json
import os
import numpy as np
import pandas as pd

st.set_page_config(page_title="DataCo Risk Predictor", layout="wide")

# 1. Path Setup:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
JSON_PATH = os.path.join(BASE_DIR, "..", "data", "interim", "category_mappings.json")

# 2. Load the models
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

# 3.UI header
st.title("DataCo Supply Chain Risk Predictor")
st.markdown("Predict delivery delays and Identify logistic risk clusters in real-time.")

# 4. Side bar for user input (The Predictor)

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

# 5. Execution Logic
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

# 6. Features from the Streamlit UI inputs
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

# 7. Button to trigger prediction and display results
if st.button("Analyze Order Risk"):
    # Supervised prediction
    prediction = model.predict(input_data)[0] # This is the YES/NO late prediction (1 for late, 0 for on-time)
    prob = model.predict_proba(input_data)[0][1] # Probability of delay

    # Unsupervised cluster assigment
    cluster = kmeans.predict(input_scaled)[0]
    readable_cluster = cluster_status.get(cluster, "Unknown Cluster")

    # Interactive UI display
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        # Calculate how much higher/lower this order is compared to the 54.73% baseline
        risk_diff = (prob - HISTORICAL_AVG_RISK) * 100
        st.metric(label= "Late Risk Probability", value= f"{prob * 100:.1f}%", delta= f"{risk_diff:.1f}% vs. History", delta_color= "inverse") # Red if risk_diff is positive, Green if negative
        
    with col2:
        if prediction == 1:
            st.metric("probability of delay:", f"{prob}")
            st.markdown("<br>", unsafe_allow_html=True) # Adds a HTML break
            st.error("Status: LATE EXPECTED")
        else:
            st.markdown("<br>", unsafe_allow_html=True) # Adds a HTML break
            st.success("Status: ON TIME")
    
    with col3:
        # Priority 1: High Risk (Probability is high)
        if prob >= 0.5:
            st.metric("Logistic Profile:", readable_cluster)
            if cluster == 2:
                st.error("Action Required: Reschedule Order")
            else:
                # This catches the Cluster 0 or 1 cases that have high probability
                st.warning("Action Recommended: Review Buffer")
        
        # Priority 2: Low Risk (Probability is low)
        else:
            st.metric("Logistic Profile:", readable_cluster)
            if (cluster == 0 or cluster == 2):
                st.success("Risk Mitigated")
            else:
                st.success("Optimal Schedule")
            
    # 8. Interactive Map for company visibility
    st.subheader("Order Location (Puerto Rico Hub)")
    # We will generate dummy coordinates based on City IDs for visual impact
    map_data = pd.DataFrame({"lat": [18.4655], # This is Puerto Rico's Center
                             "lon": [-66.1057]})
    st.map(map_data)


# 9. Dynamic Business Recommendation
    st.divider()
    st.subheader("Strategic Recommendation")

    # If the probability is safe, that is the primary recommendation
    if prob < 0.3:
        st.success("No Action Needed")
        st.write(f"The updated schedule of **{scheduled_days} day(s)** has successfully lowered the risk. This order is now safe to process.")
    
    # If the probability is high, we must suggest a change, even for "Optimal" clusters
    elif prob >= 0.5:
        if cluster == 2:
            st.error("Action Required: Reschedule Order")
            st.write(f"The current promise of **{scheduled_days} day(s)** is physically impossible for our current logistics to {selected_city}.")
            st.info(f"**To move this to 'Low Risk':** Increase 'Scheduled Days' to **4**.")
        else:
            st.warning("Action Recommended: Increase Buffer")
            st.write(f"Even though {selected_city} usually performs well, a **{scheduled_days}-day** window is too narrow for this shipping mode.")
            st.info("Recommendation: Add at least **1 additional day** to the delivery promise.")
    else:
        st.success("Optimal Parameters")
        st.write("The current parameters are efficient. No changes required.")