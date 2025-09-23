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

# Cache data loading
@st.cache_data
def load_data(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Theme toggle with vibrant colors
def set_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    if st.session_state.theme == 'dark':
        st.markdown("""
            <style>
            body { background-color: #1a1a1a; color: #f0e68c; }
            .stApp { background-color: #1a1a1a; }
            h1 { color: #ff69b4; }
            h2, h3, h4, h5, h6 { color: #00ced1; }
            .sidebar .sidebar-content { background-color: #2f2f2f; color: #ffa07a; }
            button { background-color: #ff4500; color: #ffffff; border-radius: 5px; }
            button:hover { background-color: #ff6347; color: #fffacd; }
            .stTextInput > div > div > input { background-color: #333333; color: #98fb98; }
            .stSelectbox > div > div > select { background-color: #333333; color: #dda0dd; }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body { background-color: #ffffff; color: #000000; }
            .stApp { background-color: #ffffff; }
            h1, h2, h3, h4, h5, h6 { color: #0066cc; }
            .sidebar .sidebar-content { background-color: #f0f0f0; color: #000000; }
            button { background-color: #0066cc; color: white; border-radius: 5px; }
            button:hover { background-color: #004d99; color: #ffffff; }
            </style>
            """, unsafe_allow_html=True)

# Main app
def main():
    set_theme()
    st.title(":rainbow[COVID-19 Variants Detection Analyzer]")
    st.subheader(":gray[Analyze COVID-19 Genomic Data]", divider="rainbow")

    # File uploader
    uploaded_file = st.file_uploader("Upload your COVID-19 dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.dataframe(df)
        st.info("File successfully uploaded!")

        # Sidebar navigation
        st.sidebar.title("Navigation")
        options = ["Basic Information", "Data Manipulation", "Data Visualization", "EDA", "Model Training", "Settings"]
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
            
            if st.button("Find Missing Values"):
                missing = df.isnull().sum()
                for col, count in missing.items():
                    color = "red" if count > 0 else "green"
                    st.markdown(f"<span style='color:{color}'>{col}: {count} missing (Type: {df[col].dtype})</span>", unsafe_allow_html=True)
            
            if st.button("Remove Missing Values"):
                df.dropna(inplace=True)
                st.write("Missing values removed:")
                st.dataframe(df.head())

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
            if "groupby_result" not in st.session_state:
                st.warning("Run a Group By operation first to visualize data!")
            else:
                result = st.session_state["groupby_result"]
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
                X = pd.get_dummies(df.drop(columns=["Pangolin", "Accession"], errors='ignore'))
                y = df["Pangolin"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model_type = st.selectbox("Model Type", ["Random Forest"])
                
                if st.button("Train Model"):
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Accuracy: {accuracy:.2f}")
                    st.session_state["model"] = model
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                    
                if "y_pred" in st.session_state:
                    if st.button("Show Confusion Matrix"):
                        cm = confusion_matrix(st.session_state["y_test"], st.session_state["y_pred"])
                        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                        st.plotly_chart(fig)
            else:
                st.error("Target column 'Pangolin' not found!")
                
        elif choice == "ML Advance Model":
            st.subheader(":rainbow[ML Advance Model]",divider="rainbow")
            if st.button("Cross Validation"):
                X = pd.get_dummies(df.drop(columns=["Pangolin", "Accession"], errors='ignore'))
                y = df["Pangolin"]
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scores = cross_val_score(model, X, y, cv=5)
                st.write(f"Cross-Validation Scores: {scores}")
                st.write(f"Average CV Score: {np.mean(scores):.2f}")

        # 6. Settings
        elif choice == "Settings":
            st.subheader(":rainbow[Settings]", divider="rainbow")
            if st.button("Toggle Theme"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                set_theme()
                st.experimental_rerun()

if __name__ == "__main__":
    main()

