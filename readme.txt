Hay que leer el archivo train con    
           train = pd.read_csv( AAAAA, sep= "\s+")
y ejecutar 

# Params
tree_depth = 3
timelimit = 300
n_trees = 3
Nmin = 1
c1= 15 # numero de splits activos
Format = False 
omega, omega_zero, leaf_class_sol, weights, n_nodes, train_acc, time, gap, d_sol = train_binary_wocf(train, n_trees, tree_depth, timelimit, Nmin , c1, Format)