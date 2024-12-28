import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import StandardScaler 
import optuna
import random
import shap

def optimize(file_path):
    data = pd.read_csv(file_path)
    X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
    y = data['y'] 

    def fourier_series_features(X, n_terms):  # Function to generate Fourier series features
        fourier_features = []
        for i in range(X.shape[1]):
            for n in range(1, n_terms + 1):
                fourier_features.append(np.cos(n * X.iloc[:, i]))  # Append cosine terms
                fourier_features.append(np.sin(n * X.iloc[:, i]))  # Append sine terms
        return np.column_stack(fourier_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scaling the features
    n_terms = 5  # Number of Fourier terms to include
    X_fourier = fourier_series_features(X_scaled, n_terms)

    def objective(trial):  # Objective function for Optuna optimization
        params = {  
            "objective": "reg:squarederror",  
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),  
            "max_depth": trial.suggest_int("max_depth", 3, 10),  
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),  
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),  
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)  
        }  
        model = XGBRegressor(**params)  # Creating the model with suggested parameters
        scores = cross_val_score(model, X_fourier, y, cv=5, scoring='neg_root_mean_squared_error')  
        return -scores.mean()  

    study = optuna.create_study(direction="minimize")  # Creating an Optuna study to minimize RMSE
    study.optimize(objective, n_trials=50)  # Optimizing the objective function over specified trials

    best_params = study.best_params  # Retrieving best parameters from the study
    final_model = XGBRegressor(**best_params)  # Creating final model with best parameters
    final_model.fit(X_fourier, y)  # Fitting the final model to the Fourier transformed data

    def fitness_function(individual):  
        individual_reshaped = np.array(individual).reshape(1, -1)  
        individual_fourier = fourier_series_features(scaler.transform(individual_reshaped), n_terms)  
        prediction = final_model.predict(individual_fourier)  
        return prediction[0]  

    def optimize_with_ga(num_generations=100, population_size=50):  
        num_features = X_fourier.shape[1]  # Update to use only Fourier features
        population = [np.random.uniform(-1, 1, num_features) for _ in range(population_size)]  

        for generation in range(num_generations):  
            fitnesses = [fitness_function(ind) for ind in population]  
            selected_indices = np.argsort(fitnesses)  
            selected_population = [population[i] for i in selected_indices[:population_size // 2]]  
            next_generation = []  

            while len(next_generation) < population_size:  
                parent1, parent2 = random.sample(selected_population, 2)  
                crossover_point = random.randint(1, num_features - 1)  
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))  
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))  
                next_generation.extend([child1, child2])  

            mutation_rate = 0.1  
            for individual in next_generation:  
                if random.random() < mutation_rate:  
                    mutation_index = random.randint(0, num_features - 1)  
                    individual[mutation_index] += np.random.uniform(-0.1, 0.1)  

            population = next_generation[:population_size]  

        best_index = np.argmin([fitness_function(ind) for ind in population])  
        best_individual = population[best_index]  
        best_value = fitness_function(best_individual)  

        return best_individual, best_value  

    best_solution, minimum_value = optimize_with_ga()  
    return final_model, minimum_value, best_solution  

# Best hyperparameters:  {'learning_rate': 0.052668147704993735, 'max_depth': 6, 'n_estimators': 252, 'subsample': 0.8278269745709134, 'colsample_bytree': 0.7755021309930334, 'min_child_weight': 7}
# Best RMSE:  41113349.05808047
# Best Solution (Input Values): [ 0.2707001  -0.5525737  -0.0071032   0.91410456 -0.36624142 -0.95472539
#  -0.20647895 -0.27003856  0.10552309  0.80382749  0.29506017 -0.60765564]
# Minimum Value of Approximated Function: -7653565.0

def make_equation(file_path, best_params):
    data = pd.read_csv(file_path)
    X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
    y = data['y']
    
    n_terms = 5
    
    def fourier_series_features(X, n_terms):
        fourier_features = []
        column_names = []
        for i in range(X.shape[1]):
            feature_name = f'x{i+1}'
            for n in range(1, n_terms + 1):
                cos_feature = np.cos(n * X.iloc[:, i])
                sin_feature = np.sin(n * X.iloc[:, i])
                fourier_features.append(cos_feature)
                fourier_features.append(sin_feature)
                column_names.append(f'cos({n}*{feature_name})')
                column_names.append(f'sin({n}*{feature_name})')
        return pd.DataFrame(np.column_stack(fourier_features), columns=column_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_fourier_df = fourier_series_features(pd.DataFrame(X_scaled), n_terms)
    
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_fourier_df, y)

    explainer = shap.Explainer(final_model)
    shap_values = explainer(X_fourier_df)

    def generate_function_from_shap(shap_values, feature_names):
        terms = []
        for shap_value, feature_name in zip(shap_values.values[0], feature_names):
            term = f"{shap_value:.4f} * {feature_name}"
            terms.append(term)
        return " + ".join(terms)

    function_representation = generate_function_from_shap(shap_values.values[0], X_fourier_df.columns)

    return function_representation

"""
Generated Function f(x):
-120107.9453 * cos(1*x1) + 61178.6680 * sin(1*x1) +
-57814.5469 * cos(2*x1) + -27313.0352 * sin(2*x1) +
237589.5312 * cos(3*x1) + 85096.0391 * sin(3*x1) + 
-87506.2188 * cos(4*x1) + -396947.4688 * sin(4*x1) +
9857.8477 * cos(5*x1) + -123513.9609 * sin(5*x1) + 
125825232.0000 * cos(1*x2) + -53285404.0000 * sin(1*x2) +
211994896.0000 * cos(2*x2) + -62112116.0000 * sin(2*x2) +
387617504.0000 * cos(3*x2) + -11396143.0000 * sin(3*x2) +
8563178.0000 * cos(4*x2) + -26659666.0000 * sin(4*x2) + 
1813002.2500 * cos(5*x2) + -24410554.0000 * sin(5*x2) + 
-379643.9688 * cos(1*x3) + -1547934.6250 * sin(1*x3) + 
-57385.2578 * cos(2*x3) + -86225.7266 * sin(2*x3) + 
-49677.7070 * cos(3*x3) + 52365.3555 * sin(3*x3) + 
-249729.5312 * cos(4*x3) + -277139.6250 * sin(4*x3) +
-31794.1758 * cos(5*x3) + -56249.7695 * sin(5*x3) + 
124590.4375 * cos(1*x4) + -408071.7812 * sin(1*x4) +
152046.5625 * cos(2*x4) + -199797.4531 * sin(2*x4) +
-127722.2422 * cos(3*x4) + -250297.0312 * sin(3*x4) +
-320372.7188 * cos(4*x4) + -165155.9375 * sin(4*x4) +
13993.2207 * cos(5*x4) + -155573.4375 * sin(5*x4) + 
-221461.1094 * cos(1*x5) + -235844.8125 * sin(1*x5) +
-26275.4883 * cos(2*x5) + -476841.8750 * sin(2*x5) + 
-212573.9844 * cos(3*x5) + 31979.2539 * sin(3*x5) + 
186260.6094 * cos(4*x5) + 238983.6719 * sin(4*x5) + 
-5505.8657 * cos(5*x5) + -318129.2500 * sin(5*x5) +
-62453.2266 * cos(1*x6) + -1460.8342 * sin(1*x6) + 
-10849.9785 * cos(2*x6) + -297077.1875 * sin(2*x6) +
238610.0938 * cos(3*x6) + -31840.1738 * sin(3*x6) + 
-58718.5430 * cos(4*x6) + -42830.4961 * sin(4*x6) + 
73333.1719 * cos(5)x6 )+ 71407.6094*sin ( 5* x6 )+
975 .5256*cos ( 1* x7 )+27911 .9473*sin ( 1* x7 )+
-26212 .2109*cos ( 2* x7 )+ -93362 .2812*sin ( 2* x7 )+
92296 .7500*cos ( 3* x7 )+ -24453 .2949*sin ( 3* x7 )+
116333 .6719*cos ( 4* x7 )+42791 .0469*sin ( 4* x7 )+
-129162 .7500*cos ( 5* x7 )+ -60848 .1367*sin ( 5* x7 )+
287475 .1250*cos ( 1* x8 )+ -233455 .6250*sin ( 1* x8 )+
456299 .0938*cos ( 2* x8 )+ -15742 .7734*sin ( 2* x8 )+
-65807 .1641*cos ( 3* x8 )+1850 .7922*sin ( 3* x8 )+
467392 .9688*cos ( 4* x8 )+ -134258 .6562*sin ( 4* x8 )+
-112132 .0391*cos ( 5* x8 )+80711 .9766*sin ( 5* x8 )+
44586 .1445*cos ( 1* x9 )+-40671 .2383*sin ( 1* x9 )+
44046 .7188*cos ( 2* x9 )+79660 .9375*sin ( 2* x9 )+
-767426 .6875*cos ( 3* x9 )+-45317 .2188*sin ( 3* x9 )+
-136133 .9844*cos ( 4* x9 )+-70293 .1875*sin ( 4* x9 )+
-26173 .0781*cos ( 5* x9 )+-114008 .3125*sin ( 5* x9 )+
218101 .15625*cos ( 1* x10)+-17915 .9375*sin ( 1* x10)+
-54021 .9609*cos ( 2* x10)+51770 .8906*sin ( 2* x10)+
-87357 .4219*cos ( 3* x10)+-323942 .65625*sin ( 3* x10)+
234706 .45312*cos ( 4* x10)+145518 .01562*sin ( 4* x10)+
28160 .3086*cos ( 5* x10)+339757 .71875*sin ( 5* x10)+
-33465 .16406*cos ( 1* x11)+-309676 .59375*sin ( 1* x11)+
12117 .69336*cos ( 2* x11)+27105 .179687*sin ( 2* x11)+
53359 .66406*cos ( 3* x11)+159048 .40625*sin ( 3* x11)+
-29492 .38867*cos ( 4* x11)+134751 .921875*sin ( 4* x11)+
-165476 .609375*cos ( 5* x11)+-2344 .095947*sine
"""

