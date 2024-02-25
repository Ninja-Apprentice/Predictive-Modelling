"""
OMIS6000 Assignment 2 (2024.02.23)
"""

from gurobipy import GRB
import gurobipy as gb
from gurobipy import *
import pandas as pd


"""
QUESTION 1 
"""
print()


"""
QUESTION 1 -(a): Apply the KKT conditions to find the optimal prices of each of the two computers within this series
"""
print('QUESTION1')
print("\n" + "*"*50 + "\n")  
print('QUESTION 1 -(a): Apply the KKT conditions to find the optimal prices of each of the two computers within this series')
print()

isVariablePricing = True

# Linear price response functions (intercept, slope)
response = [[35234.5457855123, 45.8964497063843], [37790.2408321369, 8.22779417263456]]

# Create a new optimization model to maximize revenue
model = gb.Model("Variable Pricing Model")

a1, b1 = response[0]
a2, b2 = response[1]

#Decision Variables
p1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p1")  # Price for Basic version
p2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p2")  # Price for Advanced version
model.setObjective(p1*(a1 - b1*p1) + p2*(a2 - b2*p2), sense=GRB.MAXIMIZE)

model.addConstr(p1 <= p2, "price_ordering_constraint")

#Constraint: Demand non-negativity 
model.addConstr(a1 - b1*p1 >= 0, "demand_nonnegativity_p1")
model.addConstr(a2 - b2*p2 >= 0, "demand_nonnegativity_p2")


# Optimize model
model.optimize()

print("Optimal price for the Basic version (p1):", p1.X)
print("Optimal price for the Advanced version (p2):", p2.X)





"""
QUESTION 1 -(b):  Optimal prices with gradient descent algorithm
"""
print("\n" + "*"*50 + "\n")  
print('QUESTION 1 -(b):  Optimal prices with gradient descent algorithm')
print()

p1 = 0
p2 = 0
step_size = 0.001
stopping_criterion = 1e-6

# create model
proj_model = gb.Model("Projected_Gradient_Descent")

# Decision Variables
p1_var = proj_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p1")
p2_var = proj_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p2")

#objective function 
objective_expr = p1*(a1 - b1*p1_var) + p2*(a2 - b2*p1_var)
proj_model.setObjective(objective_expr, GRB.MAXIMIZE)

#constraints
constraint1 = proj_model.addConstr(p1_var <= p2_var, name="price_ordering")
constraint2 = proj_model.addConstr(a1 - b1 * p1_var >= 0, name="demand_non_negativity1")
constraint3 = proj_model.addConstr(a2 - b2 * p2_var >= 0, name="demand_non_negativity2")


# Optimize model
proj_model.update()
proj_model.Params.OutputFlag = 0  # Suppress Gurobi output

while True:
    # Solve current model
    proj_model.optimize()

    # Extract current prices
    p1_curr = p1_var.X
    p2_curr = p2_var.X

    # Compute gradients
    gradient_p1 = a1 - 2 * b1 * p1_curr
    gradient_p2 = a2 - 2 * b2 * p2_curr

    # Update prices
    p1_next = max(p1_curr + step_size * gradient_p1, 0)
    p2_next = max(p2_curr + step_size * gradient_p2, 0)

    # Update variables
    p1_var.lb = p1_next
    p2_var.lb = p2_next

    # Check stopping criterion
    if max(abs(p1_next - p1_curr), abs(p2_next - p2_curr)) < stopping_criterion:
        break

    # Update current prices
    p1 = p1_next
    p2 = p2_next

# Print the result
print("Optimal prices:")
print("Optimal price for basic version p1:", p1)
print("Optimal price for advanced version p2:", p2)




"""
QUESTION 1 -(c):  Optimal revenue suggested by the mode
"""
print("\n" + "*"*50 + "\n")  
print('QUESTION 1 -(c):  Optimal revenue suggested by the mode')
print()

a = [[35234.5457855123, 37790.24083, 35675.33322],
    [37041.38038, 36846.14039, 35827.02375],
    [39414.26632, 35991.95146, 39313.31703]]

b = [[45.8964497063843, 8.22779417263456, 7.5844364095833],
    [9.03316640448659, 4.42786920644331, 2.62906001535909],
    [2.42148391836987, 4.00051240063997, 2.29662237308723]]

capacity = [[80020, 89666, 80638],
            [86740, 84050, 86565],
            [87051, 85156, 87588]]


# Create model
model_c = gb.Model("LaptopPricing")

# Decision Variable
prices = {}
for i in range(3):
    for j in range(3):
        prices[i,j] = model_c.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Price_{i}_{j}")
        
        
# Objective Function   
revenue = gb.quicksum(prices[i,j] * (a[i][j] - b[i][j] * prices[i,j]) for i in range(3) for j in range(3))
model_c.setObjective(revenue, GRB.MAXIMIZE)

# Constraints 
for i in range(3):
    for j in range(3):
        model_c.addConstr(a[i][j] - b[i][j] * prices[i,j] >= 0)
        
for i in range(3):
    for j in range(3):
        model_c.addConstr(a[i][j] - b[i][j] * prices[i,j]<= capacity[i][j])
        
for i in range(3):
    model_c.addConstr(prices[i, 0] <= prices[i, 1])
    model_c.addConstr(prices[i, 1] <= prices[i, 2])
    
# Optimize the model 
model_c.optimize()

# Print the result- Revenue
if model_c.status == GRB.OPTIMAL:
    print("Optimal Revenue: $", round(model_c.objVal, 2))
else:
    print("Optimization could not be completed.")
    
# Print the result- Optimal Price 
print("Optimal Prices:")
for i in range(3):
    for j in range(3):
        print(f"Product Line {i+1}, Version {j+1}: ${prices[i, j].X}")
        
        
        
        
        
"""
    QUESTION 1 -(d):   Optimal revenue
"""

print("\n" + "*"*50 + "\n")  
print('QUESTION 1 -(d):   Optimal revenue')
print()

# Create model
model_d = gb.Model("Pricingconstraints")

# Decision Variable
prices_d = {}
for i in range(3):
    for j in range(3):
        prices_d[i,j] = model_d.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Price_{i}_{j}")
        
# Objective Function        
revenue = gb.quicksum(prices_d[i,j] * (a[i][j] - b[i][j] * prices_d[i,j]) for i in range(3) for j in range(3))
model_d.setObjective(revenue, GRB.MAXIMIZE)


# Constraints
for i in range(3):
    for j in range(3):
        model_d.addConstr(a[i][j] - b[i][j] * prices_d[i,j] >= 0)
        
for i in range(3):
    for j in range(3):
        model_d.addConstr(a[i][j] - b[i][j] * prices_d[i,j]<= capacity[i][j])
        
for i in range(3):
    model_d.addConstr(prices_d[i, 0] <= prices_d[i, 1])
    model_d.addConstr(prices_d[i, 1] <= prices_d[i, 2])

for j in range(3):
    model_d.addConstr(prices_d[0, j] <= prices_d[1, j])
    model_d.addConstr(prices_d[1, j] <= prices_d[2, j])
    
    
# Optimize Model   
model_d.optimize()

# Print the result-Revenue
if model_d.status == GRB.OPTIMAL:
    print("Optimal Revenue: $", round(model_d.objVal, 2))
else:
    print("Optimization could not be completed.")
    
# Print the result-Optimal Price
print("Optimal Prices:")
for i in range(3):
    for j in range(3):
        print(f"Product Line {i+1}, Version {j+1}: ${prices_d[i, j].X}")
        
        
        

"""
QUESTION 2 
"""
print("\n" + "*"*50 + "\n")  
print('QUESTION 2')
print()


"""
    QUESTION 2 -(f):   What is the optimal objective function value?
        Answer: the optimal objective function value is 358 which is the optimal total of the selected player's skill score
        please see below code 
"""

print("\n" + "*"*50 + "\n")  
print('QUESTION 2 -(f):   What is the optimal objective function value')
print()
#create model
m = gb.Model("TrainingCampSelection")


# Decision Variables
invited = m.addVars(150, vtype=GRB.BINARY, name="Selected Players")

skills =pd.read_csv(r'https://raw.githubusercontent.com/Ninja-Apprentice/Modelling-and-Application/main/BasketballPlayers.csv')
skills

skills_rating = skills.iloc[:, 2:].values
skills_rating

players_positions = skills.iloc[:, 1].values
players_positions


# Objective Function
m.setObjective(quicksum(skills_rating[i][j] * invited[i] for i in range(150) for j in range(7)), GRB.MAXIMIZE)


# Constraint
m.addConstr(quicksum(invited[i] for i in range(150)) == 21) # totaly 21 players will be selected
m.addConstr(quicksum(invited[i] for i in range(150) if players_positions[i] in ['G', 'G/F']) >= 0.3 * 21)  # guards are greater than 30%
m.addConstr(quicksum(invited[i] for i in range(150) if players_positions[i] in ['F', 'C', 'F/C']) >= 0.4 * 21) #forward and center are greater than 40%

for j in range(7):# average score is greater than 2.05
    m.addConstr((1 / 21) * quicksum(skills_rating[i][j] * invited[i] for i in range(150)) >= 2.05)  
    
for i in range(19, 24): #  If any player from 20-24 (inclusive) is invited, all players from 72-78 (inclusive) cannot be
    for j in range(71, 78):
         m.addConstr(invited[i] <=1 - invited[j])

for i in range(104, 114): #  If any player from 105-114 (inclusive) is invited, at least one player from 45-49 (inclusive) and65-69 (inclusive) must be invited.
    for k in range(44, 49):
        for l in range(64, 69):
            m.addConstr(2*invited[i] <= invited[k]+invited[l])
            
for j in range(15): #t least one player must be invited from: 1 − 10, 11 − 20, 21 − 30, ..., 131 − 140, 141 − 50.
    m.addConstr(quicksum(invited[i] for i in range(j*10 , (j+1)*10)) >= 1)
    
    
# Optimize Model
m.optimize()

# Print seleted players list
selected_players = []
print("Selected players:")
for i in range(150):
    if invited[i].X > 0.5:
        selected_players.append(i)
        print(f"Player {i+1}")
        
print("\nTotal skill level of selected players:", m.objVal)



"""
    QUESTION 2 -(g):   How many guards (G, G/F) are invited to the training camp?
        Answer is 9
"""
# number of guards
num_guards = sum(1 for i in selected_players if 'G' in players_positions[i])
print("Number of guards invited:", num_guards)



"""
    QUESTION 2 -(h):   What is the smallest number of training camp invitations, and which constraint cannot be satisfied.
        Answer is 18
        The constraint "m.addConstr(quicksum(invited[i] for i in range(150)) == 21)" cannot be satisfied
"""
print("\n" + "*"*50 + "\n")  
print('QUESTION 2 -(h):   What is the smallest number of training camp invitations and constraints cannot be satisfied')
print()

# create model
m_f = gb.Model("Player_Selection")

# Decision Variable
x = m_f.addVars(150, vtype=GRB.BINARY, name="x")

# Objective Function
m_f.setObjective(quicksum(skills_rating[i][j] * x[i] for i in range(150) for j in range(7)), GRB.MAXIMIZE)
    
# Constraints
m_f.addConstr(quicksum(x[i] for i in range(150) if players_positions[i] in ['G', 'G/F']) >= 0.3 * 21)
m_f.addConstr(quicksum(x[i] for i in range(150) if players_positions[i] in ['F', 'C', 'F/C']) >= 0.4 * 21)
    
for j in range(7):
    m_f.addConstr((1 / 21) * quicksum(skills_rating[i][j] * x[i] for i in range(150)) >= 2.05)
    
for i in range(19, 24):
    for j in range(71, 78):
        m.addConstr(invited[i] <=1 - invited[j])

for i in range(104, 114):
  for k in range(44, 49):
    for l in range(64, 69):
        m_f.addConstr(2*x[i] <= x[k]+x[l])
    
for j in range(15):
    m_f.addConstr(quicksum(x[i] for i in range(j*10 , (j+1)*10)) >= 1)
    
# Solve the model iteratively
num_invitations = 150
while True:
    # Set the maximum number of solutions to 2
    m_f.setParam('PoolSolutions', 2)
    
    # Set the maximum time limit for optimization
    m_f.setParam('TimeLimit', 600)  # 10 minutes
    
    # Optimize the model
    m_f.optimize()
    
    # If the model is infeasible, print the smallest number of invitations and break the loop
    if m_f.status == gb.GRB.OPTIMAL:
        print('Optimal Objective Value:', m_f.ObjVal)
        
    elif m_f.status == gb.GRB.INFEASIBLE:
        print("Smallest number of invitations:", num_invitations)
        break
    
    # Decrease the number of invitations by 1 for the next iteration
    num_invitations -= 1
    
    # Update the constraint for the new number of invitations
    m_f.addConstr(gb.quicksum(x[i] for i in range(150)) <= num_invitations)



if m_f.status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    m_f.computeIIS()  # This computes the Irreducible Inconsistent Subsystem.
    iis_file = "model.ilp"
    m_f.write(iis_file)
    print(f"IIS written to file: {iis_file}")
    print("Review the IIS file to identify infeasible constraints.")

print()
print()
print()

# Check the constraints 
# Path to the model.ilp file
ilp_file_path = 'model.ilp'

# Open and read the contents of the model.ilp file
try:
    with open(ilp_file_path, 'r') as file:
        ilp_contents = file.read()
        print("Contents of model.ilp:")
        print(ilp_contents)
except FileNotFoundError:
    print(f"File not found: {ilp_file_path}")



"""
    QUESTION 2 -(i):   Describe (do not implement) the challenge of modifying your solution approach to ensure that
players with a total score of 12 or under would not be invited to training camp.
        
"""

print("\n" + "*"*50 + "\n")  
print('QUESTION 2 -(i):   Describe (do not implement) the challenge of modifying your solution approach')
print()

for i in range(150):
    m.addConstr(quicksum(skills_rating[i][j] * invited[i] for j in range(7)) >= 12 * invited[i])

quicksum(skills_rating[i][j] * invited[i] for j in range(7)) >= 12 * invited[i]