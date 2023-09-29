import streamlit as st
import joblib
import numpy as np
#from streamlit_lottie import st_lottie
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

st.set_page_config(
    page_title="Industrial Copper Modelling",
    page_icon="copper.png",
    layout="wide"
)

col1, col2, col3 = st.columns(3)
with col2:
    coll1,coll2,coll3 = st.columns(3)
    with coll2:
        image_path = "copper.png"
        st.image(image_path,  use_column_width="auto")
    st.write('<div style="text-align:center; font-size:40px;"><b>Industrial Copper Modelling</b></div>',
             unsafe_allow_html=True)

file_path = 'unique.txt'

# Initialize an empty list to store the numbers
options = []


def set_values(item_type, status):
    item_type_values = {
        'item type_Others': 1 if item_type == 'Others' else 0,
        'item type_PL': 1 if item_type == 'PL' else 0,
        'item type_S': 1 if item_type == 'S' else 0,
        'item type_SLAWR': 1 if item_type == 'SLAWR' else 0,
        'item type_W': 1 if item_type == 'W' else 0,
        'item type_WI': 1 if item_type == 'WI' else 0,
    }
    status_values = {
        'status_Lost': 1 if status == 'Lost' else 0,
        'status_Not lost for AM': 1 if status == 'Not lost for AM' else 0,
        'status_Offerable': 1 if status == 'Offerable' else 0,
        'status_Offered': 1 if status == 'Offered' else 0,
        'status_Revised': 1 if status == 'Revised' else 0,
        'status_To be approved': 1 if status == 'To be approved' else 0,
        'status_Won': 1 if status == 'Won' else 0,
        'status_Wonderful': 1 if status == 'Wonderful' else 0,
    }
    return item_type_values, status_values


def set_values2(item_type):
    item_type_values = {
        'item type_Others': 1 if item_type == 'Others' else 0,
        'item type_PL': 1 if item_type == 'PL' else 0,
        'item type_S': 1 if item_type == 'S' else 0,
        'item type_SLAWR': 1 if item_type == 'SLAWR' else 0,
        'item type_W': 1 if item_type == 'W' else 0,
        'item type_WI': 1 if item_type == 'WI' else 0,
    }
    return item_type_values


# Open the file for reading
with open(file_path, 'r') as file:
    # Read each line of the file
    lines = file.readlines()

    # Loop through each line and convert it to a float
    for line in lines:
        try:
            number = float(line.strip())  # Remove any leading/trailing whitespace
            options.append(number)
        except ValueError:
            print(f"Skipping invalid line: {line}")

# Now, number_list contains your numbers as a list of floats

column1, column2 = st.columns(2)
c1,c2,c3=st.columns(3)
with column1:
    with st.form("Price Prediction"):
        st.write('<div style="text-align:center; font-size:30px;">Price Prediction</div>',
                 unsafe_allow_html=True)
        # ,, ,'log_thickness', 'log_quantity_tons',
        country = st.selectbox('Country',
                               [28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0,
                                80.0, 107.0, 89.0])
        application = st.selectbox('Application',
                                   [10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0,
                                    29.0, 22.0, 40.0, 25.0, 67.0, 79.0, 3.0, 99.0, 2.0, 5.0, 39.0, 69.0, 70.0, 65.0,
                                    58.0, 68.0])
        item_type = st.selectbox('Item Type', ["W", "WI", "S", "Others", "PL", "IPL", "SLAWR"])
        status = st.selectbox('Status',
                              ["Won", "Draft", "To be approved", "Lost", "Not lost for AM", "Wonderful", "Revised",
                               "Offered", "Offerable"])
        product_ref = st.selectbox('Product Reference',
                                   [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738,
                                    1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117,
                                    1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725,
                                    1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579,
                                    929423819, 1665584320, 1665584662, 1665584642])
        Customer = st.selectbox("Customer", options)
        width = st.number_input('Width (Minimum: 1.0 and Maximum: 2990.0)', min_value=1.0, max_value=2990.0)
        thickness = st.number_input('Thickness (Minimum: 0.18 and Maximum: 400.0)', min_value=0.18,
                                    max_value=400.0)
        quantity_tons = st.number_input('Quantity in Tons (Minimum: 0.00001 and Maximum: 1000000000.0)',
                                        min_value=0.00001, max_value=1000000000.0)
        # Set values based on selected items
        item_type_values, status_values = set_values(item_type, status)
        # Store values in specified variables
        item_type_Others = item_type_values['item type_Others']
        item_type_PL = item_type_values['item type_PL']
        item_type_S = item_type_values['item type_S']
        item_type_SLAWR = item_type_values['item type_SLAWR']
        item_type_W = item_type_values['item type_W']
        item_type_WI = item_type_values['item type_WI']

        status_Lost = status_values['status_Lost']
        status_Not_lost_for_AM = status_values['status_Not lost for AM']
        status_Offerable = status_values['status_Offerable']
        status_Offered = status_values['status_Offered']
        status_Revised = status_values['status_Revised']
        status_To_be_approved = status_values['status_To be approved']
        status_Won = status_values['status_Won']
        status_Wonderful = status_values['status_Wonderful']

        submit_button = st.form_submit_button(label="Predict Price", use_container_width=True)

    if submit_button:
        model_reg = joblib.load('model_reg.joblib')
        scalar1 = joblib.load('scaler_reg.joblib')
        indepedent_Variable = np.array([[Customer, country, application, width, product_ref, np.log(thickness),
                                         np.log(quantity_tons), item_type_Others, item_type_PL, item_type_S,
                                         item_type_SLAWR, item_type_W, item_type_WI, status_Lost,
                                         status_Not_lost_for_AM, status_Offerable, status_Offered, status_Revised,
                                         status_To_be_approved, status_Won, status_Wonderful]])
        indepedent_Variable_Scaled = scalar1.transform(indepedent_Variable)
        depedent_variable = model_reg.predict(indepedent_Variable_Scaled)

        with c2.container():
            st.write(f'<div style="text-align:center; font-size:24px;">Predicted Selling Price</div>', unsafe_allow_html=True)
            result = np.exp(depedent_variable[0])
            rounded_result = round(result, 2)
            # Use HTML to change the size and center-align the text
            st.write(f'<div style="text-align:center; font-size:50px;">{rounded_result}</div>', unsafe_allow_html=True)

with column2:
    with st.form('Status Prediction'):
        st.write(f'<div style="text-align:center; font-size:30px;">Status Prediction</div>',
                 unsafe_allow_html=True)
        country2 = st.selectbox('Country',
                                [28.0, 25.0, 30.0, 32.0, 38.0, 78.0, 27.0, 77.0, 113.0, 79.0, 26.0, 39.0, 40.0, 84.0,
                                 80.0, 107.0, 89.0])
        application2 = st.selectbox('Application',
                                    [10.0, 41.0, 28.0, 59.0, 15.0, 4.0, 38.0, 56.0, 42.0, 26.0, 27.0, 19.0, 20.0, 66.0,
                                     29.0, 22.0, 40.0, 25.0, 67.0, 79.0, 3.0, 99.0, 2.0, 5.0, 39.0, 69.0, 70.0, 65.0,
                                     58.0, 68.0])
        item_type2 = st.selectbox('Item Type', ["W", "WI", "S", "Others", "PL", "IPL", "SLAWR"])
        product_ref2 = st.selectbox('Product Reference',
                                    [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738,
                                     1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117,
                                     1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725,
                                     1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579,
                                     929423819, 1665584320, 1665584662, 1665584642])
        Customer2 = st.selectbox("Customer", options)
        width2 = st.number_input('Width (Minimum: 1.0 and Maximum: 2990.0)', min_value=1.0, max_value=2990.0)
        thickness2 = st.number_input('Thickness (Minimum: 0.18 and Maximum: 400.0)', min_value=0.18,
                                     max_value=400.0)
        quantity_tons2 = st.number_input('Quantity in Tons (Minimum: 0.00001 and Maximum: 1000000000.0)',
                                         min_value=0.00001, max_value=1000000000.0)
        selling_price = st.number_input('Selling Price', min_value=0.1, max_value=81236.14)

        item_type_v = set_values2(item_type2)
        # Store values in specified variables
        item_type_Others2 = item_type_v['item type_Others']
        item_type_PL2 = item_type_v['item type_PL']
        item_type_S2 = item_type_v['item type_S']
        item_type_SLAWR2 = item_type_v['item type_SLAWR']
        item_type_W2 = item_type_v['item type_W']
        item_type_WI2 = item_type_v['item type_WI']

        submit_button2 = st.form_submit_button(label="Predict Status", use_container_width=True)
    if submit_button2:
        model_cat = joblib.load('model_Cat.joblib')
        scalar2 = joblib.load('scaler_cat.joblib')
        indepedent_Variable2 = np.array(
            [[Customer2, country2, application2, width2, product_ref2, np.log(thickness2), np.log(selling_price),
              np.log(quantity_tons2), item_type_Others2, item_type_PL2, item_type_S2,
              item_type_SLAWR2, item_type_W2, item_type_WI2]])
        indepedent_Variable2_scaled = scalar2.transform(indepedent_Variable2)
        depedent_Variable2 = model_cat.predict(indepedent_Variable2_scaled)

        with c2:
            st.write(f'<div style="text-align:center; font-size:24px;">Predicted Status</div>',
                     unsafe_allow_html=True)
            result = depedent_Variable2[0]
            # Use HTML to change the size and center-align the text
            st.write(f'<div style="text-align:center; font-size:50px;">{result}</div>', unsafe_allow_html=True)
