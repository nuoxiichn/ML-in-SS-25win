from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
airfoil_self_noise = fetch_ucirepo(id=291) 
  
# data (as pandas dataframes) 
X = airfoil_self_noise.data.features 
y = airfoil_self_noise.data.targets 
  
# metadata 
print(airfoil_self_noise.metadata) 
  
# variable information 
print(airfoil_self_noise.variables) 
