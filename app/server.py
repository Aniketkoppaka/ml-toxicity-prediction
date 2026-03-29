import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Avalon import pyAvalonTools

warnings.filterwarnings('ignore')

app = Flask(__name__)
# Enable CORS so the separate HTML file can hit this API
CORS(app)

# Cache for loading models and artifacts
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.vt = None
        self.feature_columns = None
        self.targets = None
        self.is_loaded = False
        
        # Absolute path based on server.py location -> points to CODECURE/saved_models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_dir, 'saved_models')

    def load(self):
        if self.is_loaded:
            return

        print('Loading models and artefacts...')
        # Load targets list
        self.targets = joblib.load(os.path.join(self.models_dir, 'targets.pkl'))
        
        # Load VarianceThreshold filter
        self.vt = joblib.load(os.path.join(self.models_dir, 'variance_threshold.pkl'))
        
        # Load feature columns (used by model inputs)
        self.feature_columns = joblib.load(os.path.join(self.models_dir, 'feature_columns.pkl'))
        
        # Load the 10 ensemble models
        for target in self.targets:
            safe = target.replace('-', '_')
            self.models[target] = joblib.load(os.path.join(self.models_dir, f'ensemble_{safe}.pkl'))
            
        print(f'Successfully loaded {len(self.models)} models.')
        self.is_loaded = True

registry = ModelRegistry()

def compute_features(smiles):
    """Replicates the feature engineering from the notebook for a single SMILES string.
       Uses the precise feature names from the trained VarianceThreshold model to ensure schema parity.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
        
    expected_cols = list(registry.vt.feature_names_in_)
    row_dict = {c: 0.0 for c in expected_cols}
        
    # 1. Morgan
    morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    for i, v in enumerate(morgan):
        row_dict[f'mg_{i}'] = float(v)
    
    # 2. MACCS
    maccs = list(MACCSkeys.GenMACCSKeys(mol))
    for i, v in enumerate(maccs):
        row_dict[f'mc_{i}'] = float(v)
    
    # 3. Avalon
    avalon = list(pyAvalonTools.GetAvalonFP(mol, nBits=512))
    for i, v in enumerate(avalon):
        row_dict[f'av_{i}'] = float(v)
    
    # 4. Descriptors (mapped by dict lookup if present in training schema)
    for name, fn in Descriptors.descList:
        if name in expected_cols:
            try:
                v = fn(mol)
                row_dict[name] = float(v) if (v is not None and np.isfinite(float(v))) else 0.0
            except Exception:
                row_dict[name] = 0.0
                
    # 5. Zinc properties (auto-mapped to 0.0 as they don't match training exactly)
    
    # Build dataframe directly conforming to vt expectations
    df_raw = pd.DataFrame([row_dict], columns=expected_cols)
    
    # 6. Clean logic: replace infs, fillna, clip
    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e6, 1e6)
    
    # 7. Apply VarianceThreshold filter
    X_arr = registry.vt.transform(df_clean)
    
    # 8. Create final dataframe with correct feature column names
    X_final = pd.DataFrame(X_arr, columns=registry.feature_columns)
    
    return X_final

@app.before_request
def startup():
    # Attempt to load everything on first request
    try:
        registry.load()
    except Exception as e:
        print(f"Error loading models: {e}")

@app.route('/')
def index():
    # Serve the index.html file that's in the same directory as server.py
    with open(os.path.join(os.path.dirname(__file__), 'index.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        smiles = data.get('smiles', '').strip()
        
        if not smiles:
            return jsonify({'error': 'No SMILES provided'}), 400
            
        # 1. Compute features
        X_feats = compute_features(smiles)
        
        # 2. Run predictions (with Confidence intervals across the 3 voting estimators)
        predictions = {}
        for target in registry.targets:
            clf = registry.models[target]
            
            # Since 'clf' is a sklearn VotingClassifier, we can inspect its underlying sub-estimators
            est_probs = []
            if hasattr(clf, 'estimators_'):
                for est in clf.estimators_:
                    est_probs.append(float(est.predict_proba(X_feats.values)[:, 1][0]))
            else:
                est_probs = [float(clf.predict_proba(X_feats.values)[:, 1][0])]
                
            mean_prob = float(np.mean(est_probs))
            std_prob = float(np.std(est_probs)) if len(est_probs) > 1 else 0.0
            
            predictions[target] = {
                "mean": mean_prob,
                "std": std_prob
            }
            
        return jsonify(predictions)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error during prediction'}), 500

# -------------------------------------------------------------
# FEATURE: XAI Substructure Token Highlighting
# -------------------------------------------------------------
TOXICOPHORES = {
    'Nitro': Chem.MolFromSmarts('[N+](=O)[O-]'),
    'Aniline_like': Chem.MolFromSmarts('aN'),
    'Michael_Acceptor': Chem.MolFromSmarts('C=C-C(=O)'),
    'Epoxide': Chem.MolFromSmarts('C1OC1'),
    'Thiourea': Chem.MolFromSmarts('C(=S)(N)N'),
    'Hydrazine': Chem.MolFromSmarts('NN'),
    'Alkyl_Halide_Alert': Chem.MolFromSmarts('[CX4][F,Cl,Br,I]'),
    'Polycyclic_Aromatic': Chem.MolFromSmarts('a1aa2aaaa2a1'),
    'Phenol': Chem.MolFromSmarts('c1ccccc1O'),
    'Sulfonamide': Chem.MolFromSmarts('S(=O)(=O)N')
}

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.get_json(force=True)
        smiles = data.get('smiles', '').strip()
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return jsonify({'error': 'Invalid SMILES'}), 400
            
        matches = []
        found_names = []
        
        for name, patt in TOXICOPHORES.items():
            if patt and mol.HasSubstructMatch(patt):
                found_names.append(name)
                for match_tuple in mol.GetSubstructMatches(patt):
                    matches.extend(match_tuple)
                    
        return jsonify({
            "toxicophores_found": found_names,
            "highlight_atoms": list(set(matches))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------
# FEATURE: Batch CSV Processing Endpoint
# -------------------------------------------------------------
import io
import csv
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only .csv files supported'}), 400
            
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        
        # Check if 'smiles' column exists (any case)
        headers = csv_input.fieldnames
        smiles_col = next((h for h in headers if h.lower() == 'smiles'), None)
        if not smiles_col:
            return jsonify({'error': 'CSV must contain a "smiles" column'}), 400
            
        results = []
        for row in csv_input:
            smi = row[smiles_col]
            try:
                X_feats = compute_features(smi)
                row_res = row.copy()
                for target in registry.targets:
                    clf = registry.models[target]
                    p1 = float(clf.predict_proba(X_feats.values)[:, 1][0])
                    row_res[target] = round(p1, 4)
            except Exception:
                # If valid smiles parsing fails, mark explicitly
                for target in registry.targets:
                    row_res[target] = "Error"
                    
            results.append(row_res)
            
        # Write back to CSV string
        out_stream = io.StringIO()
        out_headers = headers + registry.targets
        writer = csv.DictWriter(out_stream, fieldnames=out_headers)
        writer.writeheader()
        writer.writerows(results)
        
        # Give back as standard csv download payload
        csv_text = out_stream.getvalue()
        from flask import make_response
        response = make_response(csv_text)
        response.headers["Content-Disposition"] = "attachment; filename=toxpredict_results.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load on startup to warm cache if run directly
    registry.load()
    app.run(host='0.0.0.0', port=5000, debug=False)
