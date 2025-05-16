import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from torch_geometric.data import Data, DataLoader
import xgboost as xgb
from sklearn.svm import SVR
import tempfile
from sklearn.linear_model import LinearRegression
from GNN32noruntraining import protein_pocket_features_to_graph, ligand_features_to_graph, GNN, extract_embeddings_2
from DockedGNN import docked_protein_pocket_features_to_graph, docked_ligand_features_to_graph
from DockedGNN import GNN as GNNdocked
REFINED_MINMAX_PATH = "saved_models/min_max_values_refined.pkl"
GENERAL_MINMAX_PATH = "saved_models/general_set_model_min_max_values.pkl"

EXCEL_PATH = "GUI_affinities.xlsx"  

REFINED_GNN_PATH = "saved_models/ensemble_model_gnn.pth"
GENERAL_GNN_PATH = "saved_models/general_set_model_gnn.pth"
REFINED_XGB_PATH = "saved_models/ensemble_model_xgb.json"
GENERAL_XGB_PATH = "saved_models/general_set_model_xgb.json"
REFINED_SVM_PATH = "saved_models/ensemble_model_svm.pkl"
GENERAL_SVM_PATH = "saved_models/general_set_model_svm.pkl"
REFINED_LR_PATH = "saved_models/ensemble_model_meta.pkl"
GENERAL_LR_PATH = "saved_models/general_set_model_meta.pkl"

DOCKED_GNN_PATH = "gnn_model.pth"
DOCKED_XGB_PATH = "xgb_model.json"
DOCKED_SVM_PATH = "svm_model.pkl"
DOCKED_LR_PATH = "linear_regression_model.pkl"

st.set_page_config(page_title="Binding Affinity Predictor", layout="wide")

def home_page():
    st.markdown("<h1 style='text-align: center;'>Welcome to StructureNet's Documentation Page</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center;'>
        StructureNet is a GNN-based hybridized deep learning model built for protein-ligand binding affinity prediction. 
        This Streamlit app provides documentation on how to use StructureNet's demo page and details how the model makes graphs for training and testing. 
        Tabs in the sidebar will navigate users to graph documentation and three separate model evaluation pages. 
        Please review the citation below, which details background information on StructureNet, before navigating the webpage. 
        All code for this web application and for StructureNet's deep learning model is freely available through this 
        <a href='https://github.com/sivaGU/StructureNet' target='_blank' style='color: blue; text-decoration: underline;'>GitHub repository</a>.
    </p>
""", unsafe_allow_html=True)
    st.image("workflow.png", use_container_width=True)
    st.markdown(
        "<p style='text-align: center;'>Outline of the StructureNet Model Workflow</p>",
        unsafe_allow_html=True
    )
    st.image("graphical_abstract.jpeg", use_container_width=True)
    st.markdown(
        "<p style='text-align: center;'>Graphical Abstract</p>",
        unsafe_allow_html=True
    )
    st.write("**Citation:**")
    st.write("Kaneriya, A.; Samudrala, M.; Ganesh, H.; Moran, J.; Dandibhotla, S.; Dakshanamurthy, S. StructureNet: Physics-Informed Hybridized Deep Learning Framework for Protein–Ligand Binding Affinity Prediction. Bioengineering 2025, 12, 505.")
def get_binding_affinity(pdb_code):
    try:
        excel_data = pd.ExcelFile(EXCEL_PATH)
        for sheet_name in excel_data.sheet_names:
            sheet = excel_data.parse(sheet_name)
            if pdb_code in sheet.iloc[:, 0].values:
                return sheet.loc[sheet.iloc[:, 0] == pdb_code, sheet.columns[1]].values[0]
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    raise ValueError(f"PDB code {pdb_code} not found in the Excel file.")

def graph_explanation():
    st.markdown("<h1 style='text-align: center; font-size: 50px; font-weight: bold;'>Protein-Ligand Graph Representation Documentation</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 26px; font-weight: normal;'>This section contains information on how we create graphs of protein-ligand binding complexes for binding affinity prediction with StructureNet. There are two main steps we follow when building protein-ligand complex graphs: Feature Extraction and Graph Construction. In-depth explanations of the listed steps are provided below. </h1>", unsafe_allow_html=True)
    st.write("")
    st.image("graphrep.png", use_container_width=True) 
    st.markdown("<h1 style='text-align: center; font-size: 20px; font-weight: normal;'>An exemplary ligand molecule (viewed in PyMOL) besides the StructureNet graph created to represent it. </h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div style="text-align: center; font-size: 20px;">
        <b>Feature Extraction:</b> We start by reading in the protein and ligand molecules from their PDB files. Various file checks and sanitization procedures are performed to ensure the molecules are chemically valid and able to be easily processed. While the entire ligand molecule is represented in the ligand graph, the protein binding pocket graph only contains protein atoms within 5A of the ligand molecule. The locations of valid protein atoms within this cutoff are stored for later use in graph creation. Additionally, the locations and paths of bonds are stored to build edges in graph creation. Next, we calculate node (atom) and edge (bond) features from the processed protein binding pocket and ligand files. The specific features we used in StructureNet graphs are detailed in the citation available on the "Home Page" section of this application. All features are normalized immediately after graph creation using min-max normalization. Feature values are stored as floats in PyTorch tensors before being entered in graphs. After all node and edge features are calculated, we extract global structural features from the entire protein binding pocket and ligand molecules. A full list of the global features used in StructureNet is available in the citation under the "Home Page" section. When all features are calculated and stored in the proper data structures, we begin the graph creation process.
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div style="text-align: center; font-size: 20px;">
        <b>Graph Creation:</b> We use the NetworkX Python library to create graph structures of protein-ligand complexes. Each graph holds extracted features in node, edge, and graph-level spaces. We loop through every atom in the ligand moelcule and the valid protein binding pocket atom list to create nodes in the graph, creating a node for each atom and populating it with the respective feature array. Node feature arrays contain extracted values for atomic number, total degree, hybridization state, number of hydrogens, atomic mass, hydrogen bond donor/acceptor status (in binary), hydrophobicity index, electronegativity, element name (one-hot encoded), residue type (one-hot encoded), Voronoi regions, atomic coordinates, and spherical harmonics calculations. Next, we loop through the bonds in the ligand molecule and the protein binding pocket to create edges between nodes. Each edge is labeled with the respective bond feature array. Edge feature arays hold extracted values bond type, bond length, conjugation status, bond order, ring status, and intramolecular electrostatic interactions. for Finally, we add global features to the graph, which are stored as graph-level features. A detailed list of global molecular features is availble in the citation on the "Home Page" tab of this application. All graphs are converted to PyTorch Geometric "Data" objects for compatibility with the GNN model. This finalized graph is returned to the main algorithm for use in the StructureNet model.
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.image("featurediagram.png", use_container_width=True) 
    st.markdown("<h1 style='text-align: center; font-size: 20px; font-weight: normal;'>A visual diagram clarifying the difference in scope between node, edge, and graph-level features.</h1>", unsafe_allow_html=True)
def refined_and_general_set_testing():
    st.title("Refined and General Set Testing")

    st.write("In this section, you will be able to test StructureNet's binding affinity predictions on the PDBBBind v.2020 general and refined sets. These protein-ligand binding complexes were used when developing and testing StructureNet, so model performance will be similar to previous model performance detailed in the citation. Since the files for the refined and general sets are too large to store on this webpage, users must download them from the provided files in StructureNet's GitHub repository and manually input them into the demo application. The output from this section will give the predicted binding affinity and the experimentally-determined binding affinity from StructureNet. You can download the hydrogenated refined and general sets used to develop StructureNet from the GitHub link on the 'Home' page.")

    model_choice = st.radio("Select the model to use:", ("Refined Set", "General Set"))

    if model_choice == "Refined Set":
        gnn_path = REFINED_GNN_PATH
        xgb_path = REFINED_XGB_PATH
        svm_path = REFINED_SVM_PATH
        lr_path = REFINED_LR_PATH
        minmax_path = REFINED_MINMAX_PATH
    else:
        gnn_path = GENERAL_GNN_PATH
        xgb_path = GENERAL_XGB_PATH
        svm_path = GENERAL_SVM_PATH
        lr_path = GENERAL_LR_PATH
        minmax_path = GENERAL_MINMAX_PATH

    protein_file = st.file_uploader("Upload Protein File (.pdb)", type=["pdb"])
    ligand_file = st.file_uploader("Upload Ligand File (.pdb)", type=["pdb"])

    if protein_file and ligand_file and st.button("Predict"):
        code = protein_file.name[:4]
        st.write(f"PDB Code: {code}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_protein:
            temp_protein.write(protein_file.read())
            protein_path = temp_protein.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_ligand:
            temp_ligand.write(ligand_file.read())
            ligand_path = temp_ligand.name

        try:
            binding_affinity = get_binding_affinity(code)
            st.write(f"Retrieved Binding Affinity: {binding_affinity}")

            protein_graph = protein_pocket_features_to_graph(protein_path, ligand_path)
            ligand_graph = ligand_features_to_graph(ligand_path)

            if protein_graph is None or ligand_graph is None:
                st.error("Failed to generate the graphs. Check the input files.")
                return

            graph = protein_graph.clone()
            graph.x = torch.cat([protein_graph.x, ligand_graph.x], dim=0)
            graph.edge_index = torch.cat(
                [protein_graph.edge_index, ligand_graph.edge_index + protein_graph.x.size(0)], dim=1
            )
            graph.edge_attr = torch.cat([protein_graph.edge_attr, ligand_graph.edge_attr], dim=0)
            graph.graph_attr = torch.cat(
                [protein_graph.graph_attr.unsqueeze(0), ligand_graph.graph_attr.unsqueeze(0)], dim=1
            )
            graph.y = torch.tensor([binding_affinity], dtype=torch.float)

            st.success("Graph created successfully.")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gnn_model = GNN(hidden_channels=256, num_graph_features=10368)
            
            gnn_model.load_state_dict(torch.load(gnn_path, map_location=device))
            gnn_model.to(device)
            gnn_model.eval()

            dataloader = DataLoader([graph], batch_size=1)
            print(f"DataLoader created with {len(dataloader)} item(s).")

            # for batch in dataloader:
            #     # print("Batch Details:")
            #     # print(f"x shape: {batch.x.shape if batch.x is not None else 'None'}")
            #     # print(f"edge_index shape: {batch.edge_index.shape if batch.edge_index is not None else 'None'}")
            #     # print(f"graph_attr shape: {batch.graph_attr.shape if batch.graph_attr is not None else 'None'}")

            embeddings, _, _ = extract_embeddings_2(dataloader, gnn_model, device)

            print(type(embeddings))

            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Failed to extract embeddings. Check the input graph or GNN model.")
            
            print(type(embeddings))

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            print("Embeddings for Prediction:", embeddings)

            xgb_model = xgb.Booster()
            xgb_model.load_model(xgb_path)
            xgb_predictions = xgb_model.predict(xgb.DMatrix(embeddings))

            with open(svm_path, "rb") as f:
                svm_model = pickle.load(f)
            svr_predictions = svm_model.predict(embeddings)

            with open(lr_path, "rb") as f:
                lr_model = pickle.load(f)
            stacked_features = np.column_stack([xgb_predictions, svr_predictions])
            final_predictions = lr_model.predict(stacked_features)

            st.success(f"Predicted Binding Affinity: {final_predictions[0]:.4f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def docked_complex_testing():
    st.title("Docked Complex Testing")

    st.write("In this section, you will be able to test StructureNet's binding affinity predictions on virtually docked protein-ligand complexes to compare our performance with existing docking algorithms. Below, users can predict the binding affinity of several receptor-ligand pairs that were docked with AutoDock Vina. ")
    receptors = ["Androgen Receptor (AR)", "Chimeric Antigen Receptor (CAR)"]
    ligands_receptor_1 = ["Spironolactone", "DHT", "Testosterone", "Methyltestosterone", "Flutamide", "R1881", "Tolfenamic Acid"]
    ligands_receptor_2 = ["CINPA1", "CITCO", "PK11195", "Clotrimazole", "TO901317"]

    selected_receptor = st.selectbox("Select Receptor:", receptors)
    if selected_receptor == "Androgen Receptor (AR)":
        selected_ligand = st.selectbox("Select Ligand:", ligands_receptor_1)
    else:
        selected_ligand = st.selectbox("Select Ligand:", ligands_receptor_2)

    if selected_receptor and selected_ligand:
        if st.button("Predict Affinity"):
            st.write(f"Predicting affinity for {selected_receptor} with {selected_ligand}.")

            try:
                if selected_receptor == "Androgen Receptor (AR)":
                    if selected_ligand == "Spironolactone":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_Spironolactone/Spironolactone_out.pdb"
                        vina_pred = 6.16
                        experimental = 6.16
                    elif selected_ligand == "DHT":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_DHT/2piv_C_DHT_out.pdb"
                        vina_pred = 8.58
                        experimental = 8.65
                    elif selected_ligand == "Testosterone":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_testosterone/testosterone_out.pdb"
                        experimental = 7.80
                        vina_pred = 7.77
                    elif selected_ligand == "Methyltestosterone":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_MethylTestosterone/MethylTestosterone_out.pdb"
                        experimental = 7.80
                        vina_pred = 7.77
                    elif selected_ligand == "Flutamide":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_Flutamide/Flutamide_out.pdb"
                        experimental = 7.10
                        vina_pred = 4.11
                    elif selected_ligand == "R1881":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_R1881/R1881_out.pdb"
                        experimental = 8.51
                        vina_pred = 8.51
                    elif selected_ligand == "Tolfenamic Acid":
                        protein_file = "PFAS AR/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "PFAS AR/NewAR_TolfenamicAcid/TolfenamicAcid_out.pdb"
                        experimental = 4.33
                        vina_pred = 4.18
                elif selected_receptor == "Chimeric Antigen Receptor (CAR)":
                    if selected_ligand == "CINPA1":
                        protein_file = "PFAS CAR/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "PFAS CAR/1XNX_CINPA1/CINPA1_out.pdb"
                        experimental = 7.15
                        vina_pred = 5.87
                    elif selected_ligand == "CITCO":
                        protein_file = "PFAS CAR/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "PFAS CAR/1XNX_CITCO/CITCO_out.pdb"
                        experimental = 7.31
                        vina_pred = 6.45
                    elif selected_ligand == "Clotrimazole":
                        protein_file = "PFAS CAR/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "PFAS CAR/1XNX_clotrimazole/clotrimazole_out.pdb"
                        experimental = 6.15
                        vina_pred = 5.57
                    elif selected_ligand == "TO901317":
                        protein_file = "PFAS CAR/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "PFAS CAR/1XNX_TO901317/TO901317_out.pdb"
                        experimental = 5.66
                        vina_pred = 7.41
                    elif selected_ligand == "PK11195":
                        protein_file = "PFAS CAR/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "PFAS CAR/1XNX_PK11195/PK11195_out.pdb"
                        experimental = 6.10
                        vina_pred = 7.55



                protein_graph = docked_protein_pocket_features_to_graph(protein_file, ligand_file)
                ligand_graph = docked_ligand_features_to_graph(ligand_file)

                if protein_graph is None:
                    st.error("Protein graph generation failed.")
                    return

                if ligand_graph is None:
                    st.error("Ligand graph generation failed.")
                    return


                if protein_graph is None or ligand_graph is None:
                    st.error("Failed to generate the graphs for the selected complex.")
                    return

                graph = protein_graph.clone()
                graph.x = torch.cat([protein_graph.x, ligand_graph.x], dim=0)
                graph.edge_index = torch.cat(
                    [protein_graph.edge_index, ligand_graph.edge_index + protein_graph.x.size(0)], dim=1
                )
                graph.edge_attr = torch.cat([protein_graph.edge_attr, ligand_graph.edge_attr], dim=0)
                graph.graph_attr = torch.cat(
                    [protein_graph.graph_attr.unsqueeze(0), ligand_graph.graph_attr.unsqueeze(0)], dim=1
                )


                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                gnn_model = GNNdocked(hidden_channels=256, num_graph_features=10368)
                gnn_model.load_state_dict(torch.load(DOCKED_GNN_PATH, map_location=device))
                gnn_model.eval()

                dataloader = DataLoader([graph], batch_size=1)


                if len(dataloader) == 0:
                    st.error("Dataloader is empty. Check your input graph.")
                    return
                embeddings, _, _ = extract_embeddings_2(dataloader, gnn_model, device)

                if embeddings is None or len(embeddings) == 0:
                    st.error("Failed to extract embeddings.")
                    return

                print(type(embeddings))

                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                print("Embeddings for Prediction:", embeddings)

                xgb_model = xgb.Booster()
                xgb_model.load_model(DOCKED_XGB_PATH)
                xgb_predictions = xgb_model.predict(xgb.DMatrix(embeddings))

                with open(DOCKED_SVM_PATH, "rb") as f:
                    svm_model = pickle.load(f)
                svr_predictions = svm_model.predict(embeddings)

                with open(DOCKED_LR_PATH, "rb") as f:
                    lr_model = pickle.load(f)
                stacked_features = np.column_stack([xgb_predictions, svr_predictions])
                final_predictions = lr_model.predict(stacked_features)

                st.success(f"StructureNet Predicted Binding Affinity: {final_predictions[0]:.4f}")
                st.success(f"Vina Predicted Binding Affinity: {vina_pred}")
                st.success(f"Experimental Binding Affinity: {experimental}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

def open_testing():
    st.title("Open Testing")

    st.write("In this section, users can predict the binding affinity between protein binding pocket and ligand files of their choice using either the refined or general set models. The interface of this section appears similar to the refined and general set testing section. However, the refined and general set testing section will not allow testing on files outside of the refined and general sets - an error will be thrown. In this section, users can test the model, either from refined or general set training, on any protein-ligand complex files of their choice.")

    model_choice = st.radio("Select the trained model:", ("Refined Set", "General Set"))

    if model_choice == "Refined Set":
        gnn_path = REFINED_GNN_PATH
        xgb_path = REFINED_XGB_PATH
        svm_path = REFINED_SVM_PATH
        lr_path = REFINED_LR_PATH
        minmax_path = REFINED_MINMAX_PATH
    else:
        gnn_path = GENERAL_GNN_PATH
        xgb_path = GENERAL_XGB_PATH
        svm_path = GENERAL_SVM_PATH
        lr_path = GENERAL_LR_PATH
        minmax_path = GENERAL_MINMAX_PATH

    protein_file = st.file_uploader("Upload Protein File (.pdb)", type=["pdb"])
    ligand_file = st.file_uploader("Upload Ligand File (.pdb)", type=["pdb"])

    if protein_file and ligand_file and st.button("Predict"):


        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_protein:
            temp_protein.write(protein_file.read())
            protein_path = temp_protein.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_ligand:
            temp_ligand.write(ligand_file.read())
            ligand_path = temp_ligand.name

        try:

            protein_graph = protein_pocket_features_to_graph(protein_path, ligand_path)
            ligand_graph = ligand_features_to_graph(ligand_path)

            if protein_graph is None or ligand_graph is None:
                st.error("Failed to generate the graphs. Check the input files.")
                return

            graph = protein_graph.clone()
            graph.x = torch.cat([protein_graph.x, ligand_graph.x], dim=0)
            graph.edge_index = torch.cat(
                [protein_graph.edge_index, ligand_graph.edge_index + protein_graph.x.size(0)], dim=1
            )
            graph.edge_attr = torch.cat([protein_graph.edge_attr, ligand_graph.edge_attr], dim=0)
            graph.graph_attr = torch.cat(
                [protein_graph.graph_attr.unsqueeze(0), ligand_graph.graph_attr.unsqueeze(0)], dim=1
            )
            graph.y = torch.tensor([0], dtype=torch.float)

            st.success("Graph created successfully.")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gnn_model = GNN(hidden_channels=256, num_graph_features=10368)
            
            gnn_model.load_state_dict(torch.load(gnn_path, map_location=device))
            gnn_model.to(device)
            gnn_model.eval()

            dataloader = DataLoader([graph], batch_size=1)
            print(f"DataLoader created with {len(dataloader)} item(s).")

            # for batch in dataloader:
            #     # print("Batch Details:")
            #     # print(f"x shape: {batch.x.shape if batch.x is not None else 'None'}")
            #     # print(f"edge_index shape: {batch.edge_index.shape if batch.edge_index is not None else 'None'}")
            #     # print(f"graph_attr shape: {batch.graph_attr.shape if batch.graph_attr is not None else 'None'}")

            embeddings, _, _ = extract_embeddings_2(dataloader, gnn_model, device)

            print(type(embeddings))

            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Failed to extract embeddings. Check the input graph or GNN model.")
            
            print(type(embeddings))

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            print("Embeddings for Prediction:", embeddings)

            xgb_model = xgb.Booster()
            xgb_model.load_model(xgb_path)
            xgb_predictions = xgb_model.predict(xgb.DMatrix(embeddings))

            with open(svm_path, "rb") as f:
                svm_model = pickle.load(f)
            svr_predictions = svm_model.predict(embeddings)

            with open(lr_path, "rb") as f:
                lr_model = pickle.load(f)
            stacked_features = np.column_stack([xgb_predictions, svr_predictions])
            final_predictions = lr_model.predict(stacked_features)

            st.success(f"Predicted Binding Affinity: {final_predictions[0]:.4f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Graph Representation", "Refined and General Set Testing", "Docked Complex Testing", "Open Testing"])

if page == "Home":
    home_page()
elif page == "Graph Representation":
    graph_explanation()
elif page == "Refined and General Set Testing":
    refined_and_general_set_testing()
elif page == "Docked Complex Testing":
    docked_complex_testing()
elif page == "Open Testing":
    open_testing()
