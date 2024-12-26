
REFINED_MINMAX_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/min_max_values_refined.pkl"
GENERAL_MINMAX_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/general_set_model_min_max_values.pkl"
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

EXCEL_PATH = "/Users/arjunkaneriya/Downloads/GUI_affinities.xlsx"  

REFINED_GRAPH_PATH = "/Users/arjunkaneriya/Lombardi/saved_graphs/GNN32refinedset.pkl"
GENERAL_GRAPH_PATH = "/Users/arjunkaneriya/Lombardi/saved_graphs/GeneralSet.pkl"
REFINED_GNN_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/ensemble_model_gnn.pth"
GENERAL_GNN_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/general_set_model_gnn.pth"
REFINED_XGB_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/ensemble_model_xgb.json"
GENERAL_XGB_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/general_set_model_xgb.json"
REFINED_SVM_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/ensemble_model_svm.pkl"
GENERAL_SVM_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/general_set_model_svm.pkl"
REFINED_LR_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/ensemble_model_meta.pkl"
GENERAL_LR_PATH = "/Users/arjunkaneriya/Lombardi/saved_models/general_set_model_meta.pkl"

DOCKED_GNN_PATH = "/Users/arjunkaneriya/Downloads/gnn_model.pth"
DOCKED_XGB_PATH = "/Users/arjunkaneriya/Downloads/xgb_model.json"
DOCKED_SVM_PATH = "/Users/arjunkaneriya/Downloads/svm_model.pkl"
DOCKED_LR_PATH = "/Users/arjunkaneriya/Downloads/linear_regression_model.pkl"


st.set_page_config(page_title="Binding Affinity Predictor", layout="wide")

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

def refined_and_general_set_testing():
    st.title("Refined and General Set Testing")

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

    receptors = ["New AR", "1XNX"]
    ligands_receptor_1 = ["Spironolactone", "DHT", "Testosterone", "Methyltestosterone", "Flutamide", "R1881", "Tolfenamic Acid"]
    ligands_receptor_2 = ["CINPA1", "CITCO", "PK11195", "Clotrimazole", "TO901317"]

    selected_receptor = st.selectbox("Select Receptor:", receptors)
    if selected_receptor == "New AR":
        selected_ligand = st.selectbox("Select Ligand:", ligands_receptor_1)
    else:
        selected_ligand = st.selectbox("Select Ligand:", ligands_receptor_2)

    if selected_receptor and selected_ligand:
        if st.button("Predict Affinity"):
            st.write(f"Predicting affinity for {selected_receptor} with {selected_ligand}.")

            try:
                if selected_receptor == "New AR":
                    if selected_ligand == "Spironolactone":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_Spironolactone/Spironolactone_out.pdb"
                    elif selected_ligand == "DHT":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/2piv_C_DHT_out.pdb"
                    elif selected_ligand == "Testosterone":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_testosterone/testosterone_out.pdb"
                    elif selected_ligand == "Methyltestosterone":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_MethylTestosterone/MethylTestosterone_out.pdb"
                    elif selected_ligand == "Flutamide":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_Flutamide/Flutamide_out.pdb"
                    elif selected_ligand == "R1881":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_R1881/R1881_out.pdb"
                    elif selected_ligand == "Tolfenamic Acid":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_DHT/NewAR.pdb" 
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (Ellie)/NewAR_TolfenamicAcid/TolfenamicAcid_out.pdb"

                elif selected_receptor == "1XNX":
                    if selected_ligand == "CINPA1":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_CINPA1/CINPA1_out.pdb"
                    elif selected_ligand == "CITCO":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_CITCO/CITCO_out.pdb"
                    elif selected_ligand == "Clotrimazole":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_clotrimazole/clotrimazole_out.pdb"
                    elif selected_ligand == "TO901317":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_TO901317/TO901317_out.pdb"
                    elif selected_ligand == "PK11195":
                        protein_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_TO901317/1XNXRECEPTOR.pdb"
                        ligand_file = "/Users/arjunkaneriya/Downloads/PFAS (James)/1XNX_PK11195/PK11195_out.pdb"


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

                st.write("Graph Details:")
                st.write(f"x: {graph.x.shape if graph.x is not None else 'None'}")
                st.write(f"edge_index: {graph.edge_index.shape if graph.edge_index is not None else 'None'}")
                st.write(f"edge_attr: {graph.edge_attr.shape if graph.edge_attr is not None else 'None'}")
                st.write(f"graph_attr: {graph.graph_attr.shape if graph.graph_attr is not None else 'None'}")


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

                st.success(f"Predicted Binding Affinity: {final_predictions[0]:.4f}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

def open_testing():
    st.title("Open Testing")

    st.write("In this section, users can predict the binding affinity between protein binding pocket and ligand files of their choice using either the refined or general set models.")

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
page = st.sidebar.radio("Go to:", ["Refined and General Set Testing", "Docked Complex Testing", "Open Testing"])

if page == "Refined and General Set Testing":
    refined_and_general_set_testing()
elif page == "Docked Complex Testing":
    docked_complex_testing()
elif page == "Open Testing":
    open_testing()
