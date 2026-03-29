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
        
        # 2. Run predictions
        predictions = {}
        for target in registry.targets:
            # ensemble predict_proba returns [P(class=0), P(class=1)]
            # we want P(class=1)
            clf = registry.models[target]
            p1 = float(clf.predict_proba(X_feats.values)[:, 1][0])
            predictions[target] = p1
            
        return jsonify(predictions)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error during prediction'}), 500

if __name__ == '__main__':
    # Load on startup to warm cache if run directly
    registry.load()
    app.run(host='0.0.0.0', port=5000, debug=False)
