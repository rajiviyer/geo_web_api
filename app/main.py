# app/main.py - Streamlit app code

import streamlit as st
import pandas as pd
import numpy as np
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO
import base64
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
# import requests

# Define the FastAPI server URL
# API_URL = 'http://localhost:8000'

PLT_GRID_SHAPE = 200
DATA_DIR = "./app/data"

DATA_SELECT_OPTIONS = {
        'Use Seeded Data': 'seeded',
        'Upload New Data': 'upload'}

DATAFILE_OPTIONS = {'Temperature': 'temperature_data.csv',
                    'Soil Sand Content': 'soil_sand_content.csv'
                    }

st.set_page_config(page_title="Geospatial Interpolation",
                   page_icon="🗺️",
                   layout="wide")
def remove_top_margin():
    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=0, padding_bottom=5
        ),
        unsafe_allow_html=True
    )

def validate_selection(sel_cols: List[str]):
    if "" in sel_cols or None in sel_cols:
        return "Unselected"
    
    if len(set(sel_cols)) < 3:
        return "Duplicate"
    
    return "Fine"

### Spatial Interpolation Functions
def normalize(x:np.array)->np.array:
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    return mean, std, (x - mean)/std

def w(x:np.array, y:np.array, 
      xk:np.array, yk:np.array, 
      alpha:float=1, beta:float=2)->np.array:
    if alpha <=0 or beta<=0:
        print("alpha and beta values should be greater than 0!!")
    else:
        return (abs((x-xk)**beta) + abs((y-yk)**beta))**alpha
    
def spatial_interpol(pred_data:np.array, train_data:np.array,
                     alpha:float=1.0, beta:float=2.0, kappa:float = 2.0, 
                     eps:float=1.0e-8, delta:float = 2.2 * (1.0e-8 + 1.2 * 2))->np.array:
    
    # Rows in the training dataset
    n = train_data.shape[0]
    
    # Retrieve first 2 columns from the test dataset
    x, y = pred_data[:, 0], pred_data[:, 1]
    z = 0.0
    gamma_sum = 0.0
        
    for k in range(n):
        xk, yk, zk = train_data[k,:]
        gamma_k = 1.0
        for i in range(n):
            xi, yi, _ = train_data[i, :]
            if i!=k:
                gamma_k *= w(x, y, xi, yi, alpha, beta) / \
                        (w(xk, yk, xi, yi, alpha, beta) + eps)
                        
        dist = w(x, y, xk, yk, alpha, beta)   
        gamma_k = (eps + dist)**(-kappa) * gamma_k/(1 + gamma_k)
        gamma_k[dist > delta] = 0.0                  
        gamma_sum += gamma_k
        z += gamma_k * zk
    return z / (gamma_sum + eps)

def calculate_r_squared(y_true, y_pred):
    # Calculate the mean of the observed data
    mean_observed = np.mean(y_true)

    # Calculate the total sum of squares
    total_sum_squares = np.sum((y_true - mean_observed) ** 2)

    # Calculate the residual sum of squares
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)

    # Calculate R-squared
    r_squared = 1 - (residual_sum_squares / total_sum_squares)

    return r_squared

def filedownload(df,file_name, comment="Download csv file"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{comment}</a>'
    return href

def set_plt_params():
    # initialize visualizations
    fig = plt.figure(figsize =(4, 3), dpi=200) 
    ax = fig.gca()
    plt.setp(ax.spines.values(), linewidth=0.1)
    ax.xaxis.set_tick_params(width=0.1)
    ax.yaxis.set_tick_params(width=0.1)
    ax.xaxis.set_tick_params(length=2)
    ax.yaxis.set_tick_params(length=2)
    ax.tick_params(axis='x', labelsize=4)
    ax.tick_params(axis='y', labelsize=4)
    plt.rc('xtick', labelsize=4) 
    plt.rc('ytick', labelsize=4) 
    plt.rcParams['axes.linewidth'] = 0.1
    return(fig,ax)

# Streamlit frontend
def main():
    remove_top_margin()
    st.title('Geospatial Interpolation 🗺️')
    # open_modal = st.sidebar.button("Help")

    st.divider()
    col1, col2 = st.columns(2)
    
    data_select = st.sidebar.selectbox(
        'Select an option:',
        list(DATA_SELECT_OPTIONS.keys())
    )
    
    data_option = DATA_SELECT_OPTIONS[data_select]
    
    if data_option == "upload":
        # Upload a file
        file_upload_expander = st.sidebar.expander(":blue[**Upload Data File**] :red[*]", expanded=True)
        uploaded_file = file_upload_expander.file_uploader('File Uploader',
                                                    type=['csv'],
                                                    key="datafile",
                                                    label_visibility="hidden")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            col1.markdown("<p style='text-align: center;color:#d95a00'><b>Data Uploaded</b></p>", unsafe_allow_html=True)          
            col1.dataframe(data, height = 275, width = 500)
            columns = data.columns.tolist()
            map_columns_expander = st.sidebar.expander(":blue[**Map Columns**] ⛓️  :red[*]", expanded=True)
            select_container = map_columns_expander.container(border=True)
            x_coord = select_container.selectbox(label = "X Coordinate", 
                                        options = columns,
                                        index = None,
                                        placeholder = "Select a Column..",
                                        help = "Select Column in Uploaded Data to be considered as X Coordinate"
                                        )

            y_coord = select_container.selectbox(label = "Y Coordinate", 
                                        options = columns,
                                        index = None,
                                        placeholder = "Select a Column..",
                                        help = "Select Column in Uploaded Data to be considered as Y Coordinate"
                                        )
            
            value = select_container.selectbox(label = "Value", 
                                        options = columns,
                                        index = None,
                                        placeholder = "Select a Column..",
                                        help = "Select Column in Uploaded Data to be considered as Value"
                                        )
            
            sel_cols = [x_coord, y_coord, value]   
    else:
        seeded_filename_select = st.sidebar.selectbox(
            'Select Seeded Data:',
            list(DATAFILE_OPTIONS.keys())
        )
        
        seeded_filename = DATAFILE_OPTIONS[seeded_filename_select]
        data = pd.read_csv(f"{DATA_DIR}/{seeded_filename}")
        col1.markdown(f"<p style='text-align: center;color:#d95a00'><b>{seeded_filename_select} Data</b></p>", unsafe_allow_html=True)          
        col1.dataframe(data, height = 275, width = 500)
        columns = data.columns.tolist()        
        
    random_seed = st.sidebar.number_input(
        ":blue[**Random Seed**]  :red[*]",
        value=1040
        )
    
    action = st.sidebar.radio(":blue[**Select Action**]",
                                ["Validate", "Generate"],
                                horizontal=True,
                                help="""
                                * :orange[Validate] will split original data into train and validation sets, train the interpolation model on validation sets & will evaluate R-square. It will also display an expandable Validation Plot of Actual & Predicted Values. 
                                * :orange[Generate] will generate random locations within the space of the original dataset (for seeded data) or allow locations file upload (for uploaded data). Values for the new locations will be interpolated and the results provided as a download link. Additionally expandable interpolation plot will be displayed.
                                """)

    if action == "Generate":
        if data_option == "upload":
            location_upload_expander = st.sidebar.expander(
                ":blue[**Upload Locations File**] :red[*]", 
                expanded=True)
            
            uploaded_locations = \
                    location_upload_expander.file_uploader(
                        'Upload Locations File',
                        type=['csv'],
                        key="locations",
                        label_visibility="hidden")
                    
            if uploaded_locations is not None:
                locations_data = pd.read_csv(uploaded_locations)        
        else:
            # Get min and max values of latitude and longitude columns
            min_lat, max_lat = \
                data['latitude'].min(), data['latitude'].max()
            min_long, max_long = \
                data['longitude'].min(), data['longitude'].max()

            # Define the number of new points you want to generate
            num_points = 200  # You can adjust this number as needed

            # Initialize lists to store new latitude and longitude values
            new_latitudes = []
            new_longitudes = []

            # Generate new points until reaching the desired number
            while len(new_latitudes) < num_points:
                # Generate random latitude and longitude values within the min and max ranges
                new_lat = np.random.uniform(min_lat, max_lat)
                new_long = np.random.uniform(min_long, max_long)

                # Check if the generated pair already exists in the original DataFrame
                if not ((data['latitude'] == new_lat) & (data['longitude'] == new_long)).any():
                    new_latitudes.append(new_lat)
                    new_longitudes.append(new_long)

            # Create a new DataFrame with the generated latitude and longitude pairs
            locations_data = pd.DataFrame(
                {'longitude': new_longitudes, 
                'latitude': new_latitudes}
                )
            
            st.sidebar.success("Locations Data Generated. Click on Run to Proceed..")
    else:
        train_split = st.sidebar.slider("Train Split %",
                                        min_value=0,
                                        max_value=100,
                                        value=80
                                        )
    
    advanced_options_expander = st.sidebar.expander(":blue[**Advanced Options**]",
                                                    expanded=False)
    
    alpha = advanced_options_expander.number_input(
        ":blue[**Alpha**]  :red[*]",
        value=1.0,
        help="Small alpha increases smoothing"
        )

    beta = advanced_options_expander.number_input(
        ":blue[**Beta**]  :red[*]",
        value=2.0,
        help="Small beta increases smoothing"
        )
    
    kappa = advanced_options_expander.number_input(
        ":blue[**Kappa**]  :red[*]",
        value=2.0,
        help="High kappa makes method close to kriging"
        )
    
    delta = advanced_options_expander.number_input(
        ":blue[**Delta**]  :red[*]",
        value=2.2 * (1.0e-8 + 1.2 * 2),
        help="Used for ignoring obs too far away from sampled point"
        )

    if action == "Generate":
        countour_levels = advanced_options_expander.number_input(
        ":blue[**Coutour Levels**]  :red[*]",
        value=16,
        help="Countour Levels displayed in Interpolation Plot"
        )    

    run_button = st.sidebar.button("Run")
    if run_button:
        if data_option == "upload":
            val_sel = validate_selection(sel_cols)
        else:
            val_sel = "Fine"
            
        if val_sel == "Unselected":
            st.error("[:red[**Column Mappings**]]: Values should be Selected...", icon="🚨")
        elif val_sel == "Duplicate":
            st.error("[:red[**Column Mappings**]]: Values selected should be unique for each Selection..", icon="🚨")
        else:
            if data_option == "upload":
                # select only relevant columns from dataframe
                data = data[sel_cols]
                
            # Convert data into numpy array
            np_data = np.array(data)
                                    
            if action == "Generate":
                if locations_data.shape[1] > 2:
                    st.sidebar.error("Please upload locations with only two columns")
                else:
                    np_locations = np.array(locations_data)
                    z_pred = spatial_interpol(np_locations,
                                              np_data,
                                              alpha = alpha,
                                              beta = beta,
                                              kappa = kappa,
                                              delta = delta
                                              )
                    
                    interpolated_df = locations_data.copy()
                    interpolated_df['value'] = pd.Series(z_pred)
                    col1.markdown(
                        filedownload(interpolated_df,
                                     "interpolated_values.csv",
                                     "Download Interpolated Values CSV"
                                     ), 
                        unsafe_allow_html=True)
                    progress_text = "Generating Interpolation Plot.."
                    progress_bar = col2.progress(0, progress_text)

                    xmin = min(data.longitude)-0.01
                    xmax = max(data.longitude)+0.01

                    ymin = min(data.latitude)-0.01
                    ymax = max(data.latitude)+0.01

                    grid_x = np.linspace(xmin, xmax, PLT_GRID_SHAPE)
                    grid_y = np.linspace(ymin, ymax, PLT_GRID_SHAPE)

                    xc, yc = np.meshgrid(grid_x, grid_y)

                    zgrid = np.empty(shape=(PLT_GRID_SHAPE,
                                            PLT_GRID_SHAPE)) 
                    xg = []
                    yg = []
                    gmap = {}
                    idx = 0
                    for h in range(len(grid_x)):
                        for k in range(len(grid_y)):
                            xg.append(grid_x[h])
                            yg.append(grid_y[k])
                            gmap[h, k] = idx
                            idx += 1
                    progress_bar.progress(25, progress_text)     
                    z = spatial_interpol(np.stack((xg, yg),axis=1), 
                                         np_data,
                                         alpha = alpha,
                                         beta = beta,
                                         kappa = kappa,
                                         delta = delta   
                                         )

                    for h in range(len(grid_x)):
                        for k in range(len(grid_y)):
                            idx = gmap[h, k]
                            zgrid[h, k] = z[idx]
                            
                    zgridt = zgrid.transpose()
                    progress_bar.progress(50, progress_text)              

                    # contour plot
                    (fig, ax) = set_plt_params() 
                    cs = plt.contourf(xc, yc, 
                                      zgridt,
                                      cmap='coolwarm',
                                      levels=countour_levels)
                    
                    cs.cmap.set_under('white')
                    cs.set_clim(vmin=1)
                    cbar = plt.colorbar(cs)
                    cbar.ax.tick_params(width=0.1) 
                    cbar.ax.tick_params(length=2)
                    
                    plt.scatter(np_locations[:,0], 
                                np_locations[:,1], 
                                c=z_pred, 
                                s=8,
                                cmap=cm.coolwarm,
                                edgecolors='black',
                                linewidth=0.3,
                                alpha=0.8)                        
                    progress_bar.progress(90, progress_text) 
                    col2.markdown("<p style='text-align:center;color:#d95a00'><b>Interpolation Plot</b></p>",
                                  unsafe_allow_html = True
                                  )   
                    col2.pyplot(fig)
                    progress_bar.empty()
            else:
                if random_seed is None:
                    random_seed = 1040
                    
                np.random.seed(random_seed)
                
                # Split randomly into train and test arrays
                train_size = train_split / 100
                train_data = data.sample(frac = train_size)
                validation_data = data.drop(train_data.index)
                

                np_tr = np.array(train_data)
                np_val = np.array(validation_data)
                z_valid_actuals = np_val[:,2]
                
                # Predict on Validation Data
                z_val_predicted = spatial_interpol(np_val,
                                                   np_tr,
                                                   alpha = alpha,
                                                   beta = beta,
                                                   kappa = kappa,
                                                   delta = delta                  
                                                   )
                
                # Calculate R-Squared
                r2 = calculate_r_squared(z_valid_actuals, z_val_predicted)

                col2.markdown("<p style='text-align: center;color:#d95a00'><b>Validation Plot</b></p>", unsafe_allow_html=True)
                               
                # Plotting configuration
                sns.reset_defaults()
                plt.rcParams["figure.dpi"] = 300
                my_cmap = cm.coolwarm #mpl.colormaps['hsv'] 
                my_norm = mpl.colors.Normalize()
               # ec_colors = my_cmap(my_norm(np_tr[:,2]))
                                    
                fig, ax = plt.subplots()
                sc2a = ax.scatter(np_tr[:, 0], np_tr[:, 1], c=np_tr[:, 2], 
                                  s=7, edgecolors='black', cmap=my_cmap,
                                  linewidth=0.4, label="train")                
                sc2b = ax.scatter(np_val[:, 0], np_val[:, 1], 
                                  c=z_val_predicted, cmap=my_cmap, marker='+', 
                                  s=5, linewidth=0.4,
                                  label="validation")
                cbar1 = plt.colorbar(sc2b)
                cbar1.ax.tick_params(width=0.1)
                cbar1.ax.tick_params(length=2)
                cbar1.ax.tick_params(labelsize=10)
                plt.legend()

                col1.metric(label=":green[**Computed Validation Metrics R-squared**]📈", value=f"{r2:.3f}")

                col2.pyplot(fig) 

        
        # hide_streamlit_style = """
        #             <style>
        #             #MainMenu {visibility: hidden;}
        #             footer {visibility: hidden;}
        #             </style>
        #             """
        # st.markdown(hide_streamlit_style, unsafe_allow_html=True)         

        # Send file to the FastAPI backend for analysis
        # response = requests.post(f'{API_URL}/analyze', files={'file': uploaded_file})
