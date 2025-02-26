import mlflow
import argparse

def main(recherche_modele, optimisation_modele):
    import mlflow
    mlflow.set_experiment("MLflow Experiment Automation") 
    with mlflow.start_run():
        # Enregistrement des paramètres
        mlflow.log_param("recherche_modele", recherche_modele)
        mlflow.log_param("optimisation_modele", optimisation_modele)

        import pandas as pd
        import os

        def save_metrics_to_csv(experiment_name, run_id, params, metrics):
            """Sauvegarde les métriques dans un CSV."""
            results_dir = "mlflow_results"
            os.makedirs(results_dir, exist_ok=True)

            file_path = os.path.join(results_dir, f"{experiment_name}_metrics.csv")

            # Construire un dataframe
            df = pd.DataFrame([{**params, **metrics, "run_id": run_id}])

            # Sauvegarde en CSV (ajout si fichier existe déjà)
            df.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))
            print(f"📁 Metrics saved in {file_path}")

            # Exemple de métriques enregistrées
            params = {"recherche_modele": recherche_modele, "optimisation_modele": optimisation_modele}
            metrics = {"accuracy": 0.85, "loss": 0.15}

            save_metrics_to_csv("recherche_modele", mlflow.active_run().info.run_id, params, metrics)
  
        
            # Enregistrement d'une métrique fictive
            mlflow.log_metric("score", 0.85)
        
            print(f"Recherche modèle : {recherche_modele}, Optimisation modèle : {optimisation_modele}")

            import mlflow
            import mlflow.sklearn  # ou mlflow.pyfunc selon ton framework
            from mlflow.tracking import MlflowClient

            # Nom de l'expérience
            mlflow.set_experiment("Loan_Prediction")

            import pandas as pd
            import mlflow
            import mlflow.sklearn
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            def train_model():
            # Charger les données
            df = pd.read_csv("Loan_Data.csv")  

            # Préparer les features et la target
            X = df.drop(columns=["default"])  # Remplace "Loan_Status" par la colonne cible
            y = df["default"]

            # Séparer en train et test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialiser le modèle
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Entraîner le modèle
            model.fit(X_train, y_train)

            # Prédictions et évaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Enregistrer les métriques dans MLflow
            mlflow.log_metric("accuracy", accuracy)

            return model

           with mlflow.start_run():
           # Entraînement du modèle
           model = train_model()  # Assure-toi que cette fonction existe
    
           # Enregistrement du modèle dans MLflow
           model_uri = mlflow.sklearn.log_model(model, "loan_model")

           # Déclarer le modèle dans Model Registry
           client = MlflowClient()
           model_name = "Loan_Model"
    
           mlflow.register_model(model_uri, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recherche_modele", type=str, default="default_value")
    parser.add_argument("--optimisation_modele", type=str, default="default_value")
    args = parser.parse_args()
    
    main(args.recherche_modele, args.optimisation_modele)
