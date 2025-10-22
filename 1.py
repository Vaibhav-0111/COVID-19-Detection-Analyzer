import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

#happens

# Set page config
st.set_page_config(page_title="COVID-19  Variants Detection Analyzer", page_icon="🦠")

# Configure Streamlit to handle Arrow serialization issues
import os
# Disable Arrow optimization to avoid serialization issues
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Cache data loading
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Clean data types for Streamlit compatibility
        df = clean_dataframe_for_streamlit(df)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def clean_dataframe_for_streamlit(df):
    """Clean dataframe to be compatible with Streamlit's Arrow serialization"""
    try:
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle object columns that might cause Arrow serialization issues
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Replace NaN values first
                df_clean[col] = df_clean[col].fillna('')
                
                # Convert to string, handling special cases
                df_clean[col] = df_clean[col].astype(str)
                
                # Clean up common problematic values
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'null'], '')
                
                # Try to convert back to numeric if the data looks numeric
                try:
                    # Check if column can be converted to numeric
                    numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df_clean[col] = numeric_series
                except:
                    # Keep as string if conversion fails
                    pass
        
        # Ensure all columns have proper dtypes
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Final cleanup for remaining object columns
                df_clean[col] = df_clean[col].astype(str)
        
        return df_clean
    except Exception as e:
        st.warning(f"Data cleaning warning: {str(e)}")
        # Return original dataframe if cleaning fails
        return df

# Theme toggle with vibrant colors and animations
def set_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'  # Start in dark mode by default
    
    if st.session_state.theme == 'dark':
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
                transition: all 0.3s ease;
            }
            
            body { 
                background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
                color: #f0e68c;
                animation: fadeIn 0.8s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .stApp { 
                background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
                animation: slideIn 1s ease-out;
            }
            
            @keyframes slideIn {
                from { transform: translateX(-100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            h1 { 
                color: #ff69b4; 
                text-shadow: 0 0 20px rgba(255, 105, 180, 0.5);
                animation: glow 2s ease-in-out infinite alternate;
            }
            
            @keyframes glow {
                from { text-shadow: 0 0 20px rgba(255, 105, 180, 0.5); }
                to { text-shadow: 0 0 30px rgba(255, 105, 180, 0.8); }
            }
            
            h2, h3, h4, h5, h6 { 
                color: #00ced1; 
                text-shadow: 0 0 15px rgba(0, 206, 209, 0.4);
                animation: pulse 3s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            .sidebar .sidebar-content { 
                background: linear-gradient(180deg, #2f2f2f 0%, #1a1a1a 100%);
                color: #ffa07a;
                border-radius: 10px;
                box-shadow: 0 0 30px rgba(255, 160, 122, 0.3);
                animation: slideInLeft 0.8s ease-out;
            }
            
            @keyframes slideInLeft {
                from { transform: translateX(-100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            button { 
                background: linear-gradient(45deg, #ff4500, #ff6347);
                color: #ffffff; 
                border-radius: 25px;
                border: none;
                padding: 10px 20px;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(255, 69, 0, 0.4);
                transition: all 0.3s ease;
                animation: buttonFloat 3s ease-in-out infinite;
            }
            
            @keyframes buttonFloat {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-3px); }
            }
            
            button:hover { 
                background: linear-gradient(45deg, #ff6347, #ff4500);
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 6px 20px rgba(255, 69, 0, 0.6);
                color: #fffacd;
            }
            
            .stTextInput > div > div > input { 
                background: linear-gradient(135deg, #333333, #444444);
                color: #98fb98;
                border-radius: 10px;
                border: 2px solid transparent;
                transition: all 0.3s ease;
                animation: inputGlow 2s ease-in-out infinite alternate;
            }
            
            @keyframes inputGlow {
                from { border-color: transparent; }
                to { border-color: rgba(152, 251, 152, 0.5); }
            }
            
            .stSelectbox > div > div > select { 
                background: linear-gradient(135deg, #333333, #444444);
                color: #dda0dd;
                border-radius: 10px;
                border: 2px solid transparent;
                transition: all 0.3s ease;
            }
            
            .stDataFrame {
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 0 25px rgba(0, 206, 209, 0.3);
                animation: dataFrameSlide 0.6s ease-out;
            }
            
            @keyframes dataFrameSlide {
                from { transform: translateY(30px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            
            .stMetric {
                background: linear-gradient(135deg, rgba(0, 206, 209, 0.1), rgba(255, 105, 180, 0.1));
                border-radius: 15px;
                padding: 15px;
                border: 1px solid rgba(0, 206, 209, 0.3);
                animation: metricPulse 2s ease-in-out infinite;
            }
            
            @keyframes metricPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(90deg, #2f2f2f, #1a1a1a);
                border-radius: 10px;
                padding: 5px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: #00ced1;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(45deg, #00ced1, #ff69b4);
                color: white;
                box-shadow: 0 4px 15px rgba(0, 206, 209, 0.4);
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
                transition: all 0.3s ease;
            }
            
            body { 
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%);
                color: #000000;
                animation: fadeIn 0.8s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .stApp { 
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%);
                animation: slideIn 1s ease-out;
            }
            
            @keyframes slideIn {
                from { transform: translateX(-100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            h1, h2, h3, h4, h5, h6 { 
                color: #0066cc; 
                text-shadow: 0 0 10px rgba(0, 102, 204, 0.3);
                animation: pulse 3s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            .sidebar .sidebar-content { 
                background: linear-gradient(180deg, #f0f0f0 0%, #e9ecef 100%);
                color: #000000;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 102, 204, 0.2);
                animation: slideInLeft 0.8s ease-out;
            }
            
            @keyframes slideInLeft {
                from { transform: translateX(-100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            button { 
                background: linear-gradient(45deg, #0066cc, #004d99);
                color: white; 
                border-radius: 25px;
                border: none;
                padding: 10px 20px;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(0, 102, 204, 0.4);
                transition: all 0.3s ease;
                animation: buttonFloat 3s ease-in-out infinite;
            }
            
            @keyframes buttonFloat {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-3px); }
            }
            
            button:hover { 
                background: linear-gradient(45deg, #004d99, #0066cc);
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 6px 20px rgba(0, 102, 204, 0.6);
                color: #ffffff;
            }
            
            .stDataFrame {
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 0 20px rgba(0, 102, 204, 0.2);
                animation: dataFrameSlide 0.6s ease-out;
            }
            
            @keyframes dataFrameSlide {
                from { transform: translateY(30px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            
            .stMetric {
                background: linear-gradient(135deg, rgba(0, 102, 204, 0.1), rgba(0, 102, 204, 0.05));
                border-radius: 15px;
                padding: 15px;
                border: 1px solid rgba(0, 102, 204, 0.3);
                animation: metricPulse 2s ease-in-out infinite;
            }
            
            @keyframes metricPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(90deg, #f0f0f0, #e9ecef);
                border-radius: 10px;
                padding: 5px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: #0066cc;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(45deg, #0066cc, #004d99);
                color: white;
                box-shadow: 0 4px 15px rgba(0, 102, 204, 0.4);
            }
            </style>
            """, unsafe_allow_html=True)

# Main app
def main():
    set_theme()
    
    # Add animated header with particles effect
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
            background: linear-gradient(45deg, #ff69b4, #00ced1, #ff4500);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 3s ease-in-out infinite;
            font-size: 3rem;
            font-weight: 700;
            text-shadow: 0 0 30px rgba(255, 105, 180, 0.5);
            margin-bottom: 1rem;
        ">
            🦠 COVID-19 Variants Detection Analyzer
        </h1>
        <div style="
            background: linear-gradient(90deg, #ff69b4, #00ced1, #ff4500);
            height: 4px;
            border-radius: 2px;
            margin: 0 auto;
            width: 200px;
            animation: rainbowLine 2s ease-in-out infinite;
        "></div>
    </div>
    
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes rainbowLine {
        0% { transform: scaleX(0); }
        50% { transform: scaleX(1); }
        100% { transform: scaleX(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader(":gray[Analyze COVID-19 Genomic Data]", divider="rainbow")

    # Animated file uploader
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255, 105, 180, 0.1), rgba(0, 206, 209, 0.1));
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed rgba(255, 105, 180, 0.3);
        text-align: center;
        margin: 2rem 0;
        animation: uploaderPulse 2s ease-in-out infinite;
    ">
        <h3 style="color: #00ced1; margin-bottom: 1rem;">📁 Upload Your Dataset</h3>
        <p style="color: #ffa07a; margin-bottom: 1rem;">Drag and drop your CSV or Excel file here</p>
    </div>
    
    <style>
    @keyframes uploaderPulse {
        0%, 100% { 
            border-color: rgba(255, 105, 180, 0.3);
            box-shadow: 0 0 20px rgba(255, 105, 180, 0.2);
        }
        50% { 
            border-color: rgba(0, 206, 209, 0.5);
            box-shadow: 0 0 30px rgba(0, 206, 209, 0.4);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your COVID-19 dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Display dataframe safely to avoid Arrow serialization issues
            try:
                # Try to display with Arrow optimization disabled
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.warning(f"Display warning: {str(e)}")
                # Fallback: show dataframe info and sample
                st.write("**Dataset Information:**")
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.write("**Column Names:**")
                st.write(list(df.columns))
                st.write("**First 5 rows:**")
                st.write(df.head().to_string())
                
                # Try alternative display method
                try:
                    st.table(df.head(10))
                except:
                    st.write("**Sample Data (first 3 rows):**")
                    for i, row in df.head(3).iterrows():
                        st.write(f"Row {i}: {dict(row)}")
            
            # Animated success message
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 206, 209, 0.1));
                border-radius: 15px;
                padding: 1rem;
                border: 2px solid rgba(0, 255, 0, 0.3);
                margin: 1rem 0;
                animation: successPulse 1s ease-in-out;
            ">
                <h4 style="color: #00ff00; margin: 0; text-align: center;">
                    ✅ File Successfully Uploaded!
                </h4>
                <p style="color: #00ced1; margin: 0.5rem 0; text-align: center;">
                    📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns
                </p>
            </div>
            
            <style>
            @keyframes successPulse {{
                0% {{ transform: scale(0.8); opacity: 0; }}
                50% {{ transform: scale(1.05); opacity: 1; }}
                100% {{ transform: scale(1); opacity: 1; }}
            }}
            </style>
            """, unsafe_allow_html=True)
        else:
            st.error("Failed to load the file. Please check the file format and try again.")
            return

        # Animated sidebar navigation
        st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255, 105, 180, 0.1), rgba(0, 206, 209, 0.1));
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 2px solid rgba(255, 105, 180, 0.3);
            animation: sidebarGlow 3s ease-in-out infinite;
        ">
            <h3 style="color: #ff69b4; text-align: center; margin: 0;">🧭 Navigation</h3>
        </div>
        
        <style>
        @keyframes sidebarGlow {
            0%, 100% { 
                border-color: rgba(255, 105, 180, 0.3);
                box-shadow: 0 0 15px rgba(255, 105, 180, 0.2);
            }
            50% { 
                border-color: rgba(0, 206, 209, 0.5);
                box-shadow: 0 0 25px rgba(0, 206, 209, 0.4);
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        options = ["Basic Information", "Data Manipulation", "Data Visualization", "EDA", "Model Training", "ML Advance Model", "Settings"]
        choice = st.sidebar.selectbox("Select an Option", options)

        # 1. Basic Information
        if choice == "Basic Information":
            st.subheader(":rainbow[Basic Information]", divider="rainbow")
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Top & Bottom Rows", "Data Types", "Columns"])
            with tab1:
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                st.subheader(":gray[Statistics]", divider="gray")
                st.dataframe(df.describe())
            with tab2:
                st.subheader(":gray[Top Rows]")
                toprows = st.slider("Top rows", 1, min(df.shape[0], 50), 5, key="topslide")
                st.dataframe(df.head(toprows))
                st.subheader(":gray[Bottom Rows]")
                bottomrows = st.slider("Bottom rows", 1, min(df.shape[0], 50), 5, key="bottomslide")
                st.dataframe(df.tail(bottomrows))
            with tab3:
                st.dataframe(df.dtypes)
            with tab4:
                st.dataframe(list(df.columns))

        # 2. Data Manipulation
        elif choice == "Data Manipulation":
            st.subheader(":rainbow[Data Manipulation]", divider="rainbow")
            
            # Animated button for missing values
            st.markdown("""
            <style>
            .stButton > button {
                background: linear-gradient(45deg, #ff4500, #ff6347);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 10px 20px;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(255, 69, 0, 0.4);
                transition: all 0.3s ease;
                animation: buttonPulse 2s ease-in-out infinite;
            }
            
            @keyframes buttonPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            
            .stButton > button:hover {
                background: linear-gradient(45deg, #ff6347, #ff4500);
                transform: translateY(-2px) scale(1.1);
                box-shadow: 0 6px 20px rgba(255, 69, 0, 0.6);
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Find Missing Values"):
                missing = df.isnull().sum()
                for col, count in missing.items():
                    color = "red" if count > 0 else "green"
                    st.markdown(f"<span style='color:{color}'>{col}: {count} missing (Type: {df[col].dtype})</span>", unsafe_allow_html=True)
            
            if st.button("Remove Missing Values"):
                original_shape = df.shape
                df.dropna(inplace=True)
                new_shape = df.shape
                st.success(f"Missing values removed! Rows: {original_shape[0]} → {new_shape[0]}, Columns: {original_shape[1]} → {new_shape[1]}")
                st.dataframe(df.head())
                st.session_state["cleaned_df"] = df

            with st.expander("Group By Columns"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    groupby_cols = st.multiselect("Group by", options=list(df.columns))
                with col2:
                    operation_col = st.selectbox("Operation column", options=list(df.select_dtypes(include=[np.number]).columns) + ["Count"])
                with col3:
                    operation = st.selectbox("Operation", options=["sum", "max", "min", "mean", "median", "count"])
                
                if groupby_cols and operation_col:
                    if operation_col == "Count":
                        result = df.groupby(groupby_cols).size().reset_index(name="count")
                    else:
                        result = df.groupby(groupby_cols).agg({operation_col: operation}).reset_index()
                    st.dataframe(result)
                    st.session_state["groupby_result"] = result

        # 3. Data Visualization
        elif choice == "Data Visualization":
            st.subheader(":rainbow[Data Visualization]", divider="rainbow")
            
            # Allow visualization of either groupby results or original data
            if "groupby_result" in st.session_state:
                result = st.session_state["groupby_result"]
                st.info("Visualizing Group By results")
            else:
                result = df
                st.info("Visualizing original dataset")
            
            viz_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie", "Scatter", "Sunburst", "Heatmap", "3D Scatter"])
            
            try:
                if viz_type == "Bar":
                    x_axis = st.selectbox("X-axis", options=result.columns)
                    y_axis = st.selectbox("Y-axis", options=result.select_dtypes(include=[np.number]).columns)
                    fig = px.bar(result, x=x_axis, y=y_axis, text_auto=True, template="plotly_dark")
                    st.plotly_chart(fig)
                elif viz_type == "Line":
                    x_axis = st.selectbox("X-axis", options=result.columns)
                    y_axis = st.selectbox("Y-axis", options=result.select_dtypes(include=[np.number]).columns)
                    fig = px.line(result, x=x_axis, y=y_axis, markers=True, template="plotly_dark")
                    st.plotly_chart(fig)
                elif viz_type == "Pie":
                    names = st.selectbox("Names", options=result.columns)
                    values = st.selectbox("Values", options=result.select_dtypes(include=[np.number]).columns)
                    fig = px.pie(result, names=names, values=values, template="plotly_dark")
                    st.plotly_chart(fig)
                elif viz_type == "Scatter":
                    x_axis = st.selectbox("X-axis", options=result.columns)
                    y_axis = st.selectbox("Y-axis", options=result.select_dtypes(include=[np.number]).columns)
                    fig = px.scatter(result, x=x_axis, y=y_axis, template="plotly_dark")
                    st.plotly_chart(fig)
                elif viz_type == "Sunburst":
                    path = st.multiselect("Path", options=result.columns)
                    if path:
                        values = st.selectbox("Values", options=result.select_dtypes(include=[np.number]).columns)
                        fig = px.sunburst(result, path=path, values=values, template="plotly_dark")
                        st.plotly_chart(fig)
                elif viz_type == "Heatmap":
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        fig = px.imshow(result[numeric_cols].corr(), color_continuous_scale="RdBu", text_auto=True, template="plotly_dark")
                        st.plotly_chart(fig)
                    else:
                        st.error("Need at least 2 numeric columns for a heatmap!")
                elif viz_type == "3D Scatter":
                    x_axis = st.selectbox("X-axis", options=result.columns)
                    y_axis = st.selectbox("Y-axis", options=result.select_dtypes(include=[np.number]).columns)
                    z_axis = st.selectbox("Z-axis", options=result.select_dtypes(include=[np.number]).columns)
                    color = st.selectbox("Color", options=[None] + list(result.columns))
                    fig = px.scatter_3d(result, x=x_axis, y=y_axis, z=z_axis, color=color, template="plotly_dark")
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error rendering chart: {str(e)}")

        # 4. EDA
        elif choice == "EDA":
            st.subheader(":rainbow[Exploratory Data Analysis]", divider="rainbow")
            
            if st.button("Check Collinearity"):
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] > 1:
                    corr = numeric_df.corr()
                    fig = px.imshow(corr, color_continuous_scale="RdBu", text_auto=True, template="plotly_dark")
                    st.plotly_chart(fig)
                    threshold = 0.8
                    high_corr = [(col1, col2, corr.loc[col1, col2]) for col1 in corr.columns for col2 in corr.columns if col1 < col2 and abs(corr.loc[col1, col2]) > threshold]
                    if high_corr:
                        st.write("High collinearity pairs (>|0.8|):")
                        for col1, col2, val in high_corr:
                            st.markdown(f"<span style='color:red'>{col1} and {col2}: {val:.2f}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:green'>No high collinearity detected!</span>", unsafe_allow_html=True)
                else:
                    st.error("Not enough numeric columns for collinearity check!")
            
            if st.button("Check Outliers"):
                for col in df.select_dtypes(include=[np.number]).columns:
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=df[col], name=col, boxpoints="suspectedoutliers"))
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig)
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    st.markdown(f"<span style='color:{'red' if outliers > 0 else 'green'}'>{col}: {outliers} outliers</span>", unsafe_allow_html=True)

        # 5. Model Training
        elif choice == "Model Training":
            st.subheader(":rainbow[Model Training]", divider="rainbow")
            
            if "Pangolin" in df.columns:
                try:
                    # Prepare features and target
                    feature_cols = [col for col in df.columns if col not in ["Pangolin", "Accession"]]
                    X = pd.get_dummies(df[feature_cols])
                    y = df["Pangolin"]
                    
                    # Check if we have enough data
                    if len(X) < 10:
                        st.error("Not enough data for training! Need at least 10 samples.")
                    else:
                        # Split the data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        
                        model_type = st.selectbox("Model Type", ["Random Forest"])
                        
                        # Animated training button
                        st.markdown("""
                        <style>
                        .stButton > button {
                            background: linear-gradient(45deg, #00ced1, #ff69b4);
                            color: white;
                            border: none;
                            border-radius: 25px;
                            padding: 12px 24px;
                            font-weight: 700;
                            font-size: 16px;
                            box-shadow: 0 6px 20px rgba(0, 206, 209, 0.4);
                            transition: all 0.3s ease;
                            animation: trainButtonGlow 2s ease-in-out infinite;
                        }
                        
                        @keyframes trainButtonGlow {
                            0%, 100% { 
                                box-shadow: 0 6px 20px rgba(0, 206, 209, 0.4);
                                transform: scale(1);
                            }
                            50% { 
                                box-shadow: 0 8px 25px rgba(255, 105, 180, 0.6);
                                transform: scale(1.05);
                            }
                        }
                        
                        .stButton > button:hover {
                            background: linear-gradient(45deg, #ff69b4, #00ced1);
                            transform: translateY(-3px) scale(1.1);
                            box-shadow: 0 10px 30px rgba(255, 105, 180, 0.6);
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🚀 Train Model"):
                            with st.spinner("🤖 Training model..."):
                                if model_type == "Random Forest":
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                                
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)
                                
                                # Animated success message
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 206, 209, 0.1));
                                    border-radius: 15px;
                                    padding: 1.5rem;
                                    border: 2px solid rgba(0, 255, 0, 0.3);
                                    margin: 1rem 0;
                                    animation: successBounce 1s ease-in-out;
                                ">
                                    <h3 style="color: #00ff00; margin: 0; text-align: center;">
                                        🎉 Model Trained Successfully!
                                    </h3>
                                    <p style="color: #00ced1; margin: 0.5rem 0; text-align: center;">
                                        🤖 Random Forest Model Ready
                                    </p>
                                </div>
                                
                                <style>
                                @keyframes successBounce {{
                                    0% {{ transform: scale(0.5); opacity: 0; }}
                                    50% {{ transform: scale(1.1); opacity: 1; }}
                                    100% {{ transform: scale(1); opacity: 1; }}
                                }}
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # Animated metric
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(0, 206, 209, 0.2), rgba(255, 105, 180, 0.2));
                                    border-radius: 20px;
                                    padding: 2rem;
                                    border: 2px solid rgba(0, 206, 209, 0.5);
                                    text-align: center;
                                    margin: 1rem 0;
                                    animation: metricGlow 2s ease-in-out infinite;
                                ">
                                    <h2 style="color: #00ced1; margin: 0;">📊 Accuracy: {accuracy:.3f}</h2>
                                </div>
                                
                                <style>
                                @keyframes metricGlow {{
                                    0%, 100% {{ 
                                        border-color: rgba(0, 206, 209, 0.5);
                                        box-shadow: 0 0 20px rgba(0, 206, 209, 0.3);
                                    }}
                                    50% {{ 
                                        border-color: rgba(255, 105, 180, 0.7);
                                        box-shadow: 0 0 30px rgba(255, 105, 180, 0.5);
                                    }}
                                }}
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # Store in session state
                                st.session_state["model"] = model
                                st.session_state["X_test"] = X_test
                                st.session_state["y_test"] = y_test
                                st.session_state["y_pred"] = y_pred
                                st.session_state["feature_names"] = X.columns.tolist()
                        
                        if "y_pred" in st.session_state:
                            if st.button("Show Confusion Matrix"):
                                cm = confusion_matrix(st.session_state["y_test"], st.session_state["y_pred"])
                                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", 
                                              labels=dict(x="Predicted", y="Actual"))
                                st.plotly_chart(fig)
                                
                            if st.button("Show Feature Importance"):
                                if "model" in st.session_state:
                                    importance = st.session_state["model"].feature_importances_
                                    feature_names = st.session_state["feature_names"]
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Importance': importance
                                    }).sort_values('Importance', ascending=False)
                                    
                                    fig = px.bar(importance_df.head(20), x='Importance', y='Feature', 
                                                orientation='h', title="Top 20 Feature Importances")
                                    st.plotly_chart(fig)
                                    
                except Exception as e:
                    st.error(f"Error in model training: {str(e)}")
            else:
                st.error("Target column 'Pangolin' not found in the dataset!")
                st.info("Available columns: " + ", ".join(df.columns.tolist()))
                
        elif choice == "ML Advance Model":
            st.subheader(":rainbow[ML Advance Model]", divider="rainbow")
            
            if "Pangolin" in df.columns:
                try:
                    # Prepare features and target
                    feature_cols = [col for col in df.columns if col not in ["Pangolin", "Accession"]]
                    X = pd.get_dummies(df[feature_cols])
                    y = df["Pangolin"]
                    
                    if len(X) < 10:
                        st.error("Not enough data for cross-validation! Need at least 10 samples.")
                    else:
                        if st.button("Cross Validation"):
                            with st.spinner("Performing cross-validation..."):
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                                
                                st.success("Cross-validation completed!")
                                st.write(f"CV Scores: {[f'{score:.3f}' for score in scores]}")
                                st.metric("Average CV Score", f"{np.mean(scores):.3f}")
                                st.metric("Standard Deviation", f"{np.std(scores):.3f}")
                                
                                # Visualize CV scores
                                cv_df = pd.DataFrame({
                                    'Fold': range(1, 6),
                                    'Score': scores
                                })
                                fig = px.bar(cv_df, x='Fold', y='Score', title="Cross-Validation Scores",
                                           text='Score', template="plotly_dark")
                                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                st.plotly_chart(fig)
                                
                except Exception as e:
                    st.error(f"Error in cross-validation: {str(e)}")
            else:
                st.error("Target column 'Pangolin' not found in the dataset!")

        # 6. Settings
        elif choice == "Settings":
            st.subheader(":rainbow[Settings]", divider="rainbow")
            if st.button("Toggle Theme"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                set_theme()
                st.rerun()

if __name__ == "__main__":
    main()


# 1.ST PY

