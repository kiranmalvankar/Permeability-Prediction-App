import joblib
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from sklearn.neighbors import NearestNeighbors


# Load deployment assets

rf_model = joblib.load(
    "../Models/hybrid_rf_model_compressed.pkl"
)

reference_features = joblib.load(
    "../Models/training_reference_features_compressed.pkl"
)

ad_threshold = joblib.load(
    "../Models/final_ad_threshold.pkl"
)


# Create AD model

nn = NearestNeighbors(
    n_neighbors=2,
    metric="euclidean"
)

nn.fit(reference_features)


def smiles_to_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    fingerprint = np.array(fp)

    descriptors = np.array([

        Descriptors.MolWt(mol),

        Descriptors.MolLogP(mol),

        Descriptors.TPSA(mol),

        Descriptors.NumHDonors(mol),

        Descriptors.NumHAcceptors(mol),

        Descriptors.NumRotatableBonds(mol)

    ])

    features = np.hstack(
        [fingerprint, descriptors]
    )

    return features.reshape(1, -1)


def check_applicability_domain(features):

    distances, _ = nn.kneighbors(features)

    if distances[0][0] == 0:

        distance = distances[0][1]

    else:

        distance = distances[0][0]

    if distance <= ad_threshold:

        status = "Inside AD"
        confidence = "High"

    else:

        status = "Outside AD"
        confidence = "Low"

    return distance, status, confidence


def predict_permeability_with_ad(smiles):

    features = smiles_to_features(smiles)

    if features is None:
        return "Invalid SMILES"

    prediction = rf_model.predict(features)[0]

    distance, status, confidence = (
        check_applicability_domain(features)
    )

    return {
        "Predicted logPapp": prediction,
        "Distance": distance,
        "AD Status": status,
        "Confidence": confidence
    }