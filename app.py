import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# --- 1. DATA LOADING AND MERGING ---

st.set_page_config(layout="wide")
st.title("🚚 Predictive Delivery Optimizer")

# Load the datasets
@st.cache_data
def load_data():
    try:
        # Load data
        delivery_df = pd.read_csv("Case study internship data/delivery_performance.csv")
        orders_df = pd.read_csv("Case study internship data/orders.csv")
        routes_df = pd.read_csv("Case study internship data/routes_distance.csv")

        # --- FIX: Rename columns to be consistent and lowercase ---
        
        # Correct rename map for orders_df
        orders_rename_map = {
            'Order_ID': 'order_id',
            'Priority': 'priority_level' # <-- Fixed (was Priority_Level)
        }
        orders_df.rename(columns={k: v for k, v in orders_rename_map.items() if k in orders_df.columns}, inplace=True)
        
        # Correct rename map for delivery_df
        delivery_rename_map = {
            'Order_ID': 'order_id',
            'Promised_Delivery_Days': 'promised_delivery_time', # <-- Fixed (was Promised_Delivery_Time)
            'Actual_Delivery_Days': 'actual_delivery_time'  # <-- Fixed (was Actual_Delivery_Time)
        }
        delivery_df.rename(columns={k: v for k, v in delivery_rename_map.items() if k in delivery_df.columns}, inplace=True)

        # Correct rename map for routes_df (this one was correct)
        routes_rename_map = {
            'Order_ID': 'order_id',
            'Distance_KM': 'distance_traveled_km',
            'Traffic_Delay_Minutes': 'traffic_delays',
            'Weather_Impact': 'weather_impact'
        }
        routes_df.rename(columns={k: v for k, v in routes_rename_map.items() if k in routes_df.columns}, inplace=True)
        
        # --- END FIX ---

        # Check if the key exists before merging
        if 'order_id' not in orders_df.columns or 'order_id' not in delivery_df.columns:
            st.error("Merge failed: 'order_id' column is missing from 'orders_df' or 'delivery_df'.")
            return None 

        # Merge the dataframes into one
        merged_df = pd.merge(orders_df, delivery_df, on="order_id", how="outer")

        if 'order_id' not in routes_df.columns:
            st.error("Merge failed: 'order_id' column is missing from 'routes_df'.")
            return None 
        
        full_df = pd.merge(merged_df, routes_df, on="order_id", how="outer")
        return full_df
        
    except FileNotFoundError:
        st.error("Make sure your CSV files are in the 'Case study internship data' folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None

full_df = load_data()

if full_df is not None:
    # --- 2. FEATURE ENGINEERING ---
    
    # Convert traffic delays from minutes to hours
    if 'traffic_delays' in full_df.columns:
        full_df['traffic_delays'] = full_df['traffic_delays'] / 60.0

    # --- FIX: Updated 'is_delayed' logic ---
    # We now compare the 'Days' columns directly, which are numeric.
    # No need to convert to datetime.
    if 'actual_delivery_time' in full_df.columns and 'promised_delivery_time' in full_df.columns:
        full_df['is_delayed'] = (full_df['actual_delivery_time'] > full_df['promised_delivery_time']).astype(float)
        
        # Handle NaNs in the target variable by dropping them for the model
        model_data = full_df.dropna(subset=['is_delayed'])
        model_data['is_delayed'] = model_data['is_delayed'].astype(int)
    else:
        st.error("Could not create 'is_delayed' column. Check 'delivery_rename_map'.")
        model_data = pd.DataFrame() # Create empty dataframe to avoid more errors
    # --- END FIX ---


    # --- 3. MODEL TRAINING ---
    if not model_data.empty:
        # Define features (X) and target (y)
        required_features = ['priority_level', 'traffic_delays', 'weather_impact', 'distance_traveled_km']
        target = 'is_delayed'
        
        # Check for missing columns
        missing_cols = [col for col in required_features if col not in model_data.columns]
        if missing_cols:
            st.error(f"Error: The following required columns are missing for the model: {', '.join(missing_cols)}")
            st.warning("This is likely due to an incorrect column name in the rename mapping.")
        else:
            # Drop rows where key information for prediction is missing
            model_data = model_data.dropna(subset=required_features)

            if model_data.empty:
                st.warning("No data available for model training after dropping missing values.")
            else:
                X = model_data[required_features]
                y = model_data[target]

                # Preprocessing: Handle categorical features and missing values
                categorical_features = ['priority_level', 'weather_impact']
                numeric_features = ['traffic_delays', 'distance_traveled_km']

                # Create a preprocessor object
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', SimpleImputer(strategy='mean'), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ])

                # Create the model pipeline
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', LogisticRegression(max_iter=1000))])

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if X_train.empty:
                    st.warning("Not enough data to train the model after splitting.")
                else:
                    # Train the model
                    model.fit(X_train, y_train)

                    # Check model accuracy on the test set
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    
                    # --- 4. APP LAYOUT ---
                    st.header("Interactive Delay Predictor")
                    st.write("Use the controls below to input the details of a delivery. The model will predict the risk of a delay.")

                    # Create two columns for inputs and results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Delivery Inputs:")
                        # Create user inputs
                        priority = st.selectbox("Order Priority:", X['priority_level'].unique())
                        traffic = st.slider("Traffic Delays (in hours):", 0.0, 5.0, 1.0, 0.25)
                        weather = st.selectbox("Weather Impact:", X['weather_impact'].unique())
                        distance = st.slider("Distance (in km):", 10, 1000, 100)
                        
                        predict_button = st.button("Predict Delay Risk", type="primary")

                    with col2:
                        st.subheader("Prediction Result:")
                        # Create a button to make a prediction
                        if predict_button:
                            # Create a dataframe from user inputs
                            input_data = pd.DataFrame({
                                'priority_level': [priority],
                                'traffic_delays': [traffic],
                                'weather_impact': [weather],
                                'distance_traveled_km': [distance]
                            })

                            # Get prediction probability
                            prediction_proba = model.predict_proba(input_data)[:, 1] # Probability of being '1' (delayed)
                            delay_risk = prediction_proba[0]

                            if delay_risk > 0.5:
                                st.error(f"High Risk of Delay (Probability: {delay_risk:.0%})")
                                st.warning("Corrective Action: Consider using an express carrier or notifying the customer.")
                            else:
                                st.success(f"Low Risk of Delay (Probability: {delay_risk:.0%})")
                                st.info("No corrective action needed at this time.")

    # --- 5. DATA VISUALIZATION (MEETS REQUIREMENTS) ---
    st.divider()
    st.header("Project Analytics")
    st.write("Here are some charts that meet the project requirements and analyze our data.")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Chart 1: Bar Chart - Delay Rate by Priority
        st.subheader("Chart 1: Delay Rate by Priority Level")
        if 'priority_level' in model_data.columns:
            delay_by_priority = model_data.groupby('priority_level')['is_delayed'].mean().reset_index()
            st.bar_chart(delay_by_priority.set_index('priority_level'))
        else:
            st.warning("Missing 'priority_level' column for Chart 1.")

        # Chart 2: Scatter Plot - Distance vs. Traffic Delays
        st.subheader("Chart 2: Distance vs. Traffic Delays")
        if 'distance_traveled_km' in model_data.columns and 'traffic_delays' in model_data.columns and 'weather_impact' in model_data.columns:
            st.scatter_chart(model_data, x='distance_traveled_km', y='traffic_delays', color='weather_impact')
        else:
            st.warning("Missing columns for Chart 2.")

    with viz_col2:
        # Chart 3: Line Chart - Total Delays by Order Date
        st.subheader("Chart 3: Total Delays by Order Date")
        # --- FIX: Use 'Order_Date' from orders_df for the time chart ---
        if 'Order_Date' in model_data.columns:
            # Ensure 'Order_Date' is datetime
            model_data['Order_Date'] = pd.to_datetime(model_data['Order_Date'], errors='coerce')
            model_data_time_indexed = model_data.set_index('Order_Date')
            delays_over_time = model_data_time_indexed.resample('D')['is_delayed'].sum()
            st.line_chart(delays_over_time)
        else:
            st.warning("Missing 'Order_Date' column for Chart 3. Add 'Order_Date' to 'orders_rename_map' if it exists.")
        
        # Chart 4: Bar Chart - Weather Impact on Delays
        st.subheader("Chart 4: Impact of Weather on Delays")
        if 'weather_impact' in model_data.columns:
            weather_delays = model_data[model_data['is_delayed'] == 1]['weather_impact'].value_counts()
            st.bar_chart(weather_delays) # Bar chart is often clearer than pie
        else:
            st.warning("Missing 'weather_impact' column for Chart 4.")

    # Show the combined dataframe at the bottom
    st.divider()
    st.header("Combined Project Data")
    st.dataframe(full_df)





