import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

df = pd.read_csv("HEA_clean_dataset_with_mechanical.csv")

X = df.iloc[:, 2:64]
features = list(X.columns)

y_phase = df['Phase']
le = LabelEncoder()
y_phase_encoded = le.fit_transform(y_phase)

mech_cols = ['Young_modulus_GPa', 'Sheer_modulus_GPa', 'Bulk_modulus_GPa', 'Hardnes_Vickers_GPa']
y_mech = df[mech_cols]

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X, y_phase_encoded)

gb_reg = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
gb_reg.fit(X, y_mech)

joblib.dump(rf_clf, 'phase_model.pkl')
joblib.dump(gb_reg, 'mech_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(features, 'features.pkl')