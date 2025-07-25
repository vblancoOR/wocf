from gurobipy import *
import numpy as np
import pandas as pd
import statistics as stats


def generate_nodes(tree_depth):
    nodes = list(range(1, int(round(2 ** (tree_depth + 1)))))
    parent_nodes = nodes[0: 2 ** (tree_depth + 1) - 2 ** tree_depth - 1]
    leaf_nodes = nodes[-2 ** tree_depth:]
    return parent_nodes, leaf_nodes


def get_leaves_under_node(node_index, tree_depth):

    total_nodes = 2**(tree_depth + 1) - 1
    first_leaf_index = 2**tree_depth
    leaves = []

    def dfs(idx):
        if idx >= first_leaf_index:
            leaves.append(idx)
        else:
            dfs(2 * idx)     # left child
            dfs(2 * idx + 1) # right child

    dfs(node_index)
    return leaves


def get_parent(i,D):
    assert i > 1, "No parent for Root"
    assert i <= 2 ** (D + 1), "Error! Total: {0}; i: {1}".format(
        2 ** (D + 1), i)
    return int(i / 2)


def get_ancestors(i,D):
    assert i > 1, "No ancestors for Root"
    assert i <= 2 ** (D + 1), "Error! Total: {0}; i: {1}".format(
        2 ** (D + 1), i)
    left_ancestors = []
    right_ancestors = []
    j = i
    while j > 1:
        if j % 2 == 0:
            left_ancestors.append(int(j / 2))
        else:
            right_ancestors.append(int(j / 2))
        j = int(j / 2)
    return left_ancestors, right_ancestors


def train_binary_wocf(train, n_trees, tree_depth, timelimit, Nmin = 1,
                      c1=1, Format=False, max_epsilon = 0.0001):
    # Format = True -> class in {1,2}
    # Format = False -> class in {-1,1}

    R = range(n_trees)
    parent_nodes, leaf_nodes = generate_nodes(tree_depth)

    pp = train.shape[1] - 1
    P = range(pp)

    x = train.iloc[:,0:pp].values # x_train
    yy = train.iloc[:,-1].values # y_train


    ################ Model

    m = Model("Optimal_Binary_WForest")

    ################ Parameters

    n = x.shape[0]
    N = range(n)

    K = range(2) # range for n_classes

    if Format:
        for i in N:
            yy[i] = yy[i]-1
    else:
        for i in N:
            if yy[i] == -1:
                yy[i] = 0

    Y = np.zeros([n,2])
    for i in N:
        for k in K:
            if yy[i] == k:
                Y[i,k]=1
            else:
                Y[i,k]=0
    # Baseline accuracy
    L_hat = float(max(Y.sum(axis=0)) / n)

    parent_nodes, leaf_nodes = generate_nodes(tree_depth)

    ################ Model variables

    alpha = {}
    for i in N:
        alpha[i] = m.addVar(vtype=GRB.BINARY, name='alpha' + str(i))

    w = {}
    theta = {}
    rho = {}
    auxiliar_sym = {}
    for r in R:
        w[r] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='alpha' + str(i))
        for i in N:
            theta[r,i] = m.addVar(vtype=GRB.BINARY, name='theta' +
                                  str(r) + str(i))
            rho[r,i] = m.addVar(vtype=GRB.CONTINUOUS, name='rho' +
                                str(r) + str(i))
            auxiliar_sym[r,i] = m.addVar(vtype=GRB.BINARY, name='auxiliar_sym' +
                                         str(r) + str(i))

    z = {}
    for r in R:
        for i in N:
            for t in leaf_nodes:
                z[r,i,t] = m.addVar(vtype=GRB.BINARY, name='z' + str(r) +
                                                      str(i) + '_' + str(t))

    l = {}
    psi = {}
    for r in R:
        for t in leaf_nodes:
            l[r,t] = m.addVar(vtype=GRB.BINARY, name='l' + str(r) + str(t))
            psi[r,t] = m.addVar(vtype=GRB.BINARY, name='psi' + str(r) + str(t))

    d = {}
    for r in R:
        for t in parent_nodes:
            d[r,t] = m.addVar(vtype=GRB.BINARY, name='d' + str(r) + str(t))


    omega = {}
    for r in R:
        for j in P:
            for t in parent_nodes:
                omega[r,j,t] = m.addVar(vtype=GRB.BINARY, name='omega' +
                                        str(r) + str(j) + '_' + str(t))

    omega_zero = {}
    for r in R:
        for t in parent_nodes:
            omega_zero[r,t] = m.addVar(lb=0.0,ub=1, vtype=GRB.CONTINUOUS,
                                        name='omega_zero' + str(r) + str(t))

    ## aux_variables
    auxiliar_o = {}
    for i in N:
        auxiliar_o[i] =m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="auxliar_o" +
                                                           str(i))

    auxiliar_abs = {}
    for r in R:
        for i in N:
            for j in N:
                auxiliar_abs[r,i,j] = m.addVar(lb=0,ub=2,vtype=GRB.CONTINUOUS,
                                               name='auxiliar_ab' + str(i) +
                                               str(r))

    m.update()

    ################ Objective Function

    obj = quicksum(auxiliar_o[i] for i in N)

    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam("TimeLimit", timelimit)
    m.Params.NumericFocus = 1
    m.update()


    ################ Model Constraints


    # Objective function constraint (this controls the number of active splits)
    m.addConstr(quicksum(quicksum(d[r,t] for
                t in parent_nodes) for r in R) <= c1, name="C_objfun")

    # auxiliar constraints (0) (absolute value of errors)
    for i in N:
        m.addConstr(yy[i]-alpha[i] <= auxiliar_o[i], name="Caux00a[%d]"%(i+1))
        m.addConstr(-yy[i]+alpha[i] <= auxiliar_o[i], name="Caux00b[%d]"%(i+1))


    # Breaking tree symmetries
    for i in N:
        for r in R:
            m.addConstr(yy[i]-theta[r,i] <= auxiliar_sym[r,i], name="CauxST1[%d,%d]"%(r+1,i+1))
            m.addConstr(-yy[i]+theta[r,i]<= auxiliar_sym[r,i], name="CauxST2[%d,%d]"%(r+1,i+1))
            m.addConstr(auxiliar_sym[r,i]<= yy[i]+theta[r,i], name="CauxST3[%d,%d]"%(r+1,i+1))
            m.addConstr(auxiliar_sym[r,i]<= 2-yy[i]+theta[r,i], name="CauxST4[%d,%d]"%(r+1,i+1))

    for r in R:
        for j in R:
            if j > r:
                m.addConstr(quicksum(auxiliar_sym[r,i] for i in N )<=
                            quicksum(auxiliar_sym[j,i] for i in N) , name="C_sym[%d,%d]"%(r+1,j+1))


    ### C1
    m.addConstr(quicksum(w[r] for r in R)==1, name="weights constraint")

    ### C2
    for i in N:
        m.addConstr(quicksum(rho[r,i] for r in R) - 0.5 + 0.001 <= alpha[i],
                                                    name="C02a[%d]"%(i+1))
        m.addConstr(-quicksum(rho[r,i] for r in R) + 0.5 <= 1-alpha[i],
                                                    name="C02b[%d]"%(i+1))

    #
    for r in R:
        for i in N:
            m.addConstr(w[r] - (1-theta[r,i]) <= rho[r,i])
            m.addConstr(rho[r,i] <= w[r] + (1-theta[r,i]))
            m.addConstr(rho[r,i] <= theta[r,i])

    ### C3
    for r in R:
        for i in N:
            for t in leaf_nodes:
                m.addConstr(psi[r,t] - (1-z[r,i,t]) <= theta[r,i] )
                m.addConstr( theta[r,i] <= psi[r,t] + (1-z[r,i,t]) )

    ### C4 - C5 - c6
    for r in R:
        for t in parent_nodes:
            m.addConstr(quicksum(omega[r,j,t] for j in P) == d[r,t],
                                        name="C04[%d,%d]"%(r+1,t+1))
            m.addConstr(omega_zero[r,t] <= d[r,t],
                                        name="C05[%d,%d]"%(r+1,t+1))
            if t != 1:
                m.addConstr(d[r,t] <= d[r,get_parent(t,tree_depth)],
                                        name="C06[%d,%d]"%(r+1,t+1))

    ### C7 - C8
    for r in R:
        for i in N:
            m.addConstr(quicksum(z[r,i,t] for t in leaf_nodes) == 1,
                                        name="C07[%d,%d]"%(r+1,i+1))
            for t in leaf_nodes:
                m.addConstr(z[r,i,t] <= l[r,t],
                                name="C08[%d,%d,%d]"%(r+1,i+1,t+1))

    ### C9
    for r in R:
        for t in leaf_nodes:
            m.addConstr(quicksum(z[r,i,t] for i in N) >= l[r,t] * Nmin,
                                        name="C09[%d,%d]"%(r+1,t+1))

    ### C10 - C11
    for r in R:
        for i in N:
            for t in leaf_nodes:
                left_ancestors, right_ancestors = get_ancestors(t,tree_depth)
                for mm in right_ancestors:
                    m.addConstr(
                        quicksum(omega[r,j,mm] * x[i, j] for j in P) >=
                        omega_zero[r,mm] - (1 - z[r,i,t]),
                                        name="C10[%d,%d,%d]"%(r+1,i+1,t+1))
                for mm in left_ancestors:
                    m.addConstr(
                        quicksum(omega[r,j, mm] * (x[i, j]) for j in P)+max_epsilon
                        <= omega_zero[r,mm] +
                        (1 - z[r,i,t]) * (1+max_epsilon),
                                        name="C11[%d,%d,%d]"%(r+1,i+1,t+1))

    m.update()

    m.optimize()

    gap = m.MIPGap
    time = m.Runtime
    if m.Status == 3:
            m.computeIIS()
            m.write('infeasible_constraints.ilp')

    # Saving the solution

    def modify_leaf_class(leaf_class_aux, list_d, tree_depth):
        num_internal_nodes = 2**tree_depth -1
        num_leaves = 2**tree_depth

        for r in R:
            for leaf in range(num_leaves):
                next_idx = leaf + num_internal_nodes + 1
                while leaf_class_aux[r,leaf] == -1:
                    next_idx = get_parent(next_idx, tree_depth)
                    leaves = get_leaves_under_node(next_idx, tree_depth)
                    leaves_idx = [x - num_internal_nodes - 1 for x in leaves]
                    leaf_class_aux[r,leaf] = np.max(leaf_class_aux[r, leaves_idx])

        return leaf_class_aux
    

    if (m.SolCount>0 and m.ObjVal< 1e+100):
        n_nodes = 0
        omega_sol = np.zeros((n_trees, pp,len(parent_nodes)))
        omega_zero_sol = np.zeros((n_trees,len(parent_nodes)))
        theta_sol = np.zeros((n_trees,n))
        leaf_class_sol = np.zeros((n_trees,len(leaf_nodes)))
        list_d = np.zeros((n_trees,len(parent_nodes)))
        w_sol = np.zeros(n_trees)
        for r in R:
            w_sol[r] = w[r].x
            for t in parent_nodes:
                n_nodes = n_nodes + d[r,t].x
                list_d[r,t-1]=d[r,t].x
                for p in P:
                    omega_sol[r,p,t-1]= omega[r,p,t].x

                omega_zero_sol[r,t-1] = omega_zero[r,t].x
            for t in leaf_nodes:
                class_l=-1
                if l[r,t].x > 0.5:
                    for i in N:
                        if z[r,i,t].x>0.5:
                            class_l = abs(theta[r,i].x)
                leaf_class_sol[r,t-len(parent_nodes)-1]=class_l

        leaf_class_sol = modify_leaf_class(leaf_class_sol,list_d,tree_depth)

    errors = 0
    for i in N:
        errors += auxiliar_o[i].x
    train_accuracy = 1-errors/n

    return(omega_sol, omega_zero_sol, leaf_class_sol, w_sol, n_nodes, train_accuracy, time, gap, list_d)
